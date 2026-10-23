import numpy as np
import torch

from facies_ensembles.latent.decode import decode_volume
from facies_ensembles.latent.trajectories import sample_trajectories
from .update import es_mda_step
from .wells import extract_well_labels, reference_observations, sample_well_ij

def run_esmda(
    generator,
    *,
    rule,
    n_ens,
    depth,
    latent_shape,
    reference_volume,
    n_wells=4,
    n_assimilations=4,
    alpha=None,
    obs_variance=0.01,
    localization=None,
    seed=0,
    device="cpu",
    rule_kwargs=None,
    scale=None,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    h, w = int(reference_volume.shape[1]), int(reference_volume.shape[2])
    wells = sample_well_ij(n_wells, h, w, rng=rng)
    z_obs = reference_observations(np.asarray(reference_volume), wells)
    alphas = list(alpha) if alpha is not None else [float(n_assimilations)] * int(n_assimilations)

    z = sample_trajectories(
        rule,
        n_ens,
        depth,
        latent_shape,
        device=device,
        **(rule_kwargs or {}),
    )

    m = z.reshape(n_ens, -1).detach().cpu().numpy().astype(np.float64).T
    cd = np.eye(z_obs.shape[0], dtype=np.float64) * float(obs_variance)
    history = []

    for a in alphas:
        z_batch = torch.from_numpy(
            m.T.reshape(n_ens, depth, *latent_shape).astype(np.float32)
        ).to(device)
        labels = decode_volume(generator, z_batch, scale=scale)
        pred = extract_well_labels(labels.numpy(), wells)
        m = es_mda_step(m, z_obs, pred, a, cd, corr=localization)
        history.append(
            {
                "alpha": float(a),
                "mismatch": float(np.mean((pred - z_obs) ** 2)),
            }
        )

    z_post = torch.from_numpy(
        m.T.reshape(n_ens, depth, *latent_shape).astype(np.float32)
    )
    labels_post = decode_volume(generator, z_post.to(device), scale=scale)

    return {
        "wells": wells,
        "z_prior": z.detach().cpu(),
        "z_post": z_post,
        "labels_post": labels_post,
        "history": history,
        "z_obs": z_obs,
    }
