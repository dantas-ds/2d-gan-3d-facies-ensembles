import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from facies_ensembles.faciesgan.build import build_discriminator, build_generator
from facies_ensembles.faciesgan.checkpoint import save_checkpoint
from facies_ensembles.data.h5_dataset import ScaleH5Dataset
from .losses import discriminator_loss, generator_loss, gradient_penalty

class ProgressiveTrainer:
    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() and cfg.get("gpu", True) else "cpu")
        )
        self.n_classes = int(cfg["n_classes"])
        self.latent_shape = list(cfg.get("latent_shape", [4, 4, 4]))
        self.depths = list(cfg.get("depths", [128, 128, 128, 128, 64, 32]))
        self.extras = list(cfg.get("extras", [0, 1, 1, 1, 1, 2]))
        self.kernels = list(cfg.get("kernels", [3, 3, 3, 3, 3, 3]))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.lambda_gp = float(cfg.get("lambda_gp", 10.0))
        self.epsilon_d = float(cfg.get("epsilon_d", 0.001))
        self.n_critic = int(cfg.get("n_critic", 1))
        self.log_every = int(cfg.get("log_every", 500))
        self.save_every = int(cfg.get("save_every", 10000))
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.net_g = build_generator(
            self.n_classes,
            latent_shape=self.latent_shape,
            depths=self.depths,
            extras=self.extras,
            kernels=self.kernels,
        ).to(self.device)
        self.net_d = build_discriminator(
            self.n_classes,
            latent_shape=self.latent_shape,
            depths=self.depths,
            extras=self.extras,
            kernels=self.kernels,
            fc_depth=int(cfg.get("fc_depth", 256)),
        ).to(self.device)
        self.opt_g = torch.optim.Adam(
            self.net_g.parameters(), lr=float(cfg.get("lr", 0.0015)), betas=(0.0, 0.99)
        )
        self.opt_d = torch.optim.Adam(
            self.net_d.parameters(), lr=float(cfg.get("lr", 0.0015)), betas=(0.0, 0.99)
        )
        self.scale_iters = list(cfg.get("max_iter_at_scale", [160000] * len(self.depths)))
        self.lr_schedule = {
            int(k): float(v) for k, v in (cfg.get("lr_schedule") or {}).items()
        }
        self.h5_path = cfg["data_h5"]
        self.alpha_n_jumps = list(cfg.get("alpha_n_jumps", [0] + [5000] * (len(self.depths) - 1)))
        self.alpha_size_jumps = list(
            cfg.get("alpha_size_jumps", [0] + [32] * (len(self.depths) - 1))
        )

    def _meta(self):
        return {
            "n_classes": self.n_classes,
            "latent_shape": self.latent_shape,
            "depths": self.depths,
            "extras": self.extras,
            "kernels": self.kernels,
            "fc_depth": int(self.cfg.get("fc_depth", 256)),
        }

    def _set_lr(self, scale):
        lr = self.lr_schedule.get(scale, float(self.cfg.get("lr", 0.0015)))

        for opt in (self.opt_g, self.opt_d):
            for group in opt.param_groups:
                group["lr"] = lr

    def _loader(self, scale):
        ds = ScaleH5Dataset(self.h5_path, scale, n_classes=self.n_classes, one_hot=True)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 2)),
            drop_last=True,
            pin_memory=self.device.type == "cuda",
        )

    def _alpha(self, scale, it):
        if scale <= 0:
            return 0.0

        n_jumps = int(self.alpha_n_jumps[scale])
        size = int(self.alpha_size_jumps[scale])
        fade = n_jumps * size

        if fade <= 0:
            return 0.0

        return max(0.0, 1.0 - float(it) / float(fade))

    def train(self, start_scale=0, end_scale=None):
        end_scale = len(self.depths) - 1 if end_scale is None else int(end_scale)
        history = []

        for scale in range(int(start_scale), end_scale + 1):
            self._set_lr(scale)
            loader = self._loader(scale)
            it_loader = iter(loader)
            n_iter = int(self.scale_iters[scale])

            for it in range(n_iter):
                alpha = self._alpha(scale, it)
                loss_d = 0.0

                for _ in range(self.n_critic):
                    try:
                        real = next(it_loader)
                    except StopIteration:
                        it_loader = iter(loader)
                        real = next(it_loader)

                    real = real.to(self.device)
                    z = torch.randn(real.size(0), *self.latent_shape, device=self.device)
                    fake = self.net_g(z, scale=scale, alpha=alpha).detach()

                    self.opt_d.zero_grad(set_to_none=True)

                    real_score = self.net_d(real, scale=scale, alpha=alpha)
                    fake_score = self.net_d(fake, scale=scale, alpha=alpha)
                    gp = gradient_penalty(
                        self.net_d,
                        real,
                        fake,
                        scale=scale,
                        alpha=alpha,
                        weight=self.lambda_gp,
                    )
                    d_loss = discriminator_loss(
                        real_score, fake_score, drift=self.epsilon_d
                    ) + gp
                    d_loss.backward()
                    self.opt_d.step()
                    loss_d = float(d_loss.detach())

                try:
                    real = next(it_loader)
                except StopIteration:
                    it_loader = iter(loader)
                    real = next(it_loader)

                real = real.to(self.device)
                z = torch.randn(real.size(0), *self.latent_shape, device=self.device)

                self.opt_g.zero_grad(set_to_none=True)

                fake = self.net_g(z, scale=scale, alpha=alpha)
                g_loss = generator_loss(self.net_d(fake, scale=scale, alpha=alpha))
                g_loss.backward()
                self.opt_g.step()
                loss_g = float(g_loss.detach())

                if it % self.log_every == 0 or it + 1 == n_iter:
                    row = {
                        "scale": scale,
                        "iter": it,
                        "alpha": alpha,
                        "loss_g": loss_g,
                        "loss_d": loss_d,
                    }
                    history.append(row)
                    print(
                        f"scale={scale} iter={it} alpha={alpha:.4f} "
                        f"G={loss_g:.4f} D={loss_d:.4f}",
                        flush=True,
                    )

                if (it + 1) % self.save_every == 0 or it + 1 == n_iter:
                    ckpt = self.out_dir / f"s{scale}_i{it + 1:07d}.pt"
                    save_checkpoint(
                        ckpt,
                        generator=self.net_g,
                        discriminator=self.net_d,
                        meta={**self._meta(), "scale": scale, "iter": it + 1},
                    )

        with open(self.out_dir / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f)

        return history
