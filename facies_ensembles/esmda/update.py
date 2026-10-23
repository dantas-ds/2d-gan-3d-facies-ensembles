import math

import numpy as np

def es_mda_step(m_ens, z_obs, pred_ens, alpha, cd, corr=None, numsave=2):
    num_ens = m_ens.shape[1]
    varn = 1.0 - 1.0 / math.pow(10, numsave)
    ym = m_ens.mean(axis=1, keepdims=True)
    dm = pred_ens.mean(axis=1, keepdims=True)
    dmf = m_ens - ym
    ddf = pred_ens - dm

    cmd = np.dot(dmf, ddf.T) / (num_ens - 1)
    cdd = np.dot(ddf, ddf.T) / (num_ens - 1)

    r = np.linalg.cholesky(cd)
    u_noise = r.T

    aux = np.repeat(z_obs, num_ens, axis=1)
    noise = np.random.multivariate_normal(
        np.zeros(z_obs.shape[0]), np.eye(z_obs.shape[0]), num_ens
    ).T
    d_obs = aux + math.sqrt(float(alpha)) * np.dot(u_noise, noise)

    u, s, vh = np.linalg.svd(cdd + float(alpha) * cd)
    v = vh.T
    diagonal = s
    cut = len(diagonal) - 1

    for i in range(len(diagonal)):
        if (sum(diagonal[0 : i + 1])) / (sum(diagonal)) > varn:
            cut = i
            break

    u = u[:, 0 : cut + 1]
    v = v[:, 0 : cut + 1]
    ss = np.diag(diagonal[0 : cut + 1] ** (-1))
    k = np.dot(cmd, np.dot(np.dot(v, ss), u.T))

    if corr is not None and len(corr) > 0:
        k = corr * k

    return m_ens + np.dot(k, (d_obs - pred_ens))
