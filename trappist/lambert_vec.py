"""
Vectorised (NumPy) zero-revolution Lambert solver (Izzo 2015).

Solves many Lambert problems that share the departure point r1 but have
different arrival points / times of flight in one call. It is used only in the
search phase (tree search), where thousands of candidate legs are scored;
final solutions are always re-evaluated with the exact scalar solver used by
the fitness function, so small differences here never reach the results.
"""
import numpy as np


def _x2tof(x, lam):
    """Non-dimensional time of flight T(x), N = 0 (Lancaster + Battin series)."""
    E = x * x - 1.0
    rho = np.abs(E)
    z = np.sqrt(1.0 + lam * lam * E)
    dist = np.abs(x - 1.0)
    out = np.empty_like(x)

    near = dist < 0.01                         # Battin series around x = 1
    if np.any(near):
        xn, ln, zn = x[near], lam[near], z[near]
        eta = zn - ln * xn
        S1 = 0.5 * (1.0 - ln - xn * eta)
        Q = np.ones_like(S1)
        C = np.ones_like(S1)
        for j in range(40):
            C = C * (3.0 + j) * (1.0 + j) / (2.5 + j) * S1 / (j + 1)
            Q = Q + C
        Q = 4.0 / 3.0 * Q
        out[near] = (eta ** 3 * Q + 4.0 * ln * eta) / 2.0

    far = ~near                                 # Lancaster expression
    if np.any(far):
        xf, lf, zf, Ef = x[far], lam[far], z[far], E[far]
        y = np.sqrt(np.abs(Ef))
        g = xf * zf - lf * Ef
        d = np.where(Ef < 0,
                     np.arccos(np.clip(g, -1.0, 1.0)),
                     np.log(np.maximum(y * (zf - lf * xf) + g, 1e-300)))
        out[far] = (xf - lf * zf - d / y) / Ef
    return out


def lambert_batch(r1, r2, tof, mu, iters=12):
    """r1: (3,), r2: (n,3), tof: (n,) seconds -> v1 (n,3), v2 (n,3), ok (n,)."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    tof = np.asarray(tof, dtype=float)
    n = r2.shape[0]
    c = r2 - r1
    cn = np.linalg.norm(c, axis=1)
    R1 = np.linalg.norm(r1)
    R2 = np.linalg.norm(r2, axis=1)
    s = (cn + R1 + R2) / 2.0
    ir1 = np.broadcast_to(r1 / R1, (n, 3))
    ir2 = r2 / R2[:, None]
    ih = np.cross(ir1, ir2)
    ih /= np.linalg.norm(ih, axis=1)[:, None]
    lam2 = 1.0 - cn / s
    lam = np.sqrt(np.maximum(lam2, 0.0))
    neg = ih[:, 2] < 0.0
    lam = np.where(neg, -lam, lam)
    it1 = np.where(neg[:, None], np.cross(ir1, ih), np.cross(ih, ir1))
    it2 = np.where(neg[:, None], np.cross(ir2, ih), np.cross(ih, ir2))
    lam3 = lam * lam2
    T = np.sqrt(2.0 * mu / s ** 3) * tof

    T00 = np.arccos(lam) + lam * np.sqrt(1.0 - lam2)
    T1 = 2.0 / 3.0 * (1.0 - lam3)
    with np.errstate(all="ignore"):
        x = np.where(T >= T00, -(T - T00) / (T - T00 + 4.0),
            np.where(T <= T1, T1 * (T1 - T) / (2.0 / 5.0 * (1.0 - lam2 * lam3) * T) + 1.0,
                     (T / T00) ** (0.69314718055994529 / np.log(T1 / T00)) - 1.0))
        for _ in range(iters):
            tx = _x2tof(x, lam)
            umx2 = 1.0 - x * x
            y = np.sqrt(1.0 - lam2 * umx2)
            DT = (3.0 * tx * x - 2.0 + 2.0 * lam3 * x / y) / umx2
            DDT = (3.0 * tx + 5.0 * x * DT + 2.0 * (1.0 - lam2) * lam3 / y ** 3) / umx2
            DDDT = (7.0 * x * DDT + 8.0 * DT - 6.0 * (1.0 - lam2) * lam2 * lam3 * x / y ** 5) / umx2
            delta = tx - T
            DT2 = DT * DT
            step = delta * (DT2 - delta * DDT / 2.0) / (DT * (DT2 - delta * DDT) + DDDT * delta * delta / 6.0)
            x = x - np.where(np.isfinite(step), step, 0.0)
            x = np.maximum(x, -0.999999)        # stay on the physical branch
        gamma = np.sqrt(mu * s / 2.0)
        rho = (R1 - R2) / cn
        sigma = np.sqrt(1.0 - rho * rho)
        y = np.sqrt(1.0 - lam2 + lam2 * x * x)
        vr1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / R1
        vr2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / R2
        vt = gamma * sigma * (y + lam * x)
        v1 = vr1[:, None] * ir1 + (vt / R1)[:, None] * it1
        v2 = vr2[:, None] * ir2 + (vt / R2)[:, None] * it2
        resid = np.abs(_x2tof(x, lam) - T) / T
    ok = np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1) & (resid < 1e-8)
    return v1, v2, ok
