"""
Vectorised (NumPy) versions of the routines needed to evaluate ONE LEG of the
official encoding for many parameter sets at once:

    fly-by (beta, rp)  ->  coast eta*T  ->  DSM  ->  Lambert (1-eta)*T  ->  planet j

Used by the leg optimiser inside the beam search. The arithmetic is the same
as in astro.py / problem.py (checked in tests/test_vec.py).
"""
import numpy as np

from .astro import DAY2SEC, EARTH_RADIUS
from .problem import MU_TRAPPIST, SAFE_RADIUS_TRAPPIST

MU = MU_TRAPPIST


def _norm(a):
    return np.sqrt(np.einsum("ij,ij->i", a, a))


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


# ----------------------------------------------------------------- fly-by
def fb_prop_batch(v_in, v_pla, rp_ratio, beta, mu):
    vrel = np.asarray(v_in) - np.asarray(v_pla)
    vn = np.linalg.norm(vrel)
    i_hat = vrel / vn
    j_hat = np.cross(i_hat, v_pla)
    j_hat /= np.linalg.norm(j_hat)
    k_hat = np.cross(i_hat, j_hat)
    ecc = 1.0 + rp_ratio * EARTH_RADIUS / mu * vn * vn
    delta = 2.0 * np.arcsin(1.0 / ecc)
    c1 = vn * np.cos(delta)
    c2 = vn * np.cos(beta) * np.sin(delta)
    c3 = vn * np.sin(beta) * np.sin(delta)
    return v_pla + c1[:, None] * i_hat + c2[:, None] * j_hat + c3[:, None] * k_hat


# ----------------------------------------------------------------- propagation
def _stumpff(z):
    C = np.empty_like(z)
    S = np.empty_like(z)
    pos, neg, zer = z > 1e-8, z < -1e-8, np.abs(z) <= 1e-8
    sz = np.sqrt(z[pos])
    C[pos] = (1 - np.cos(sz)) / z[pos]
    S[pos] = (sz - np.sin(sz)) / sz ** 3
    sz = np.sqrt(-z[neg])
    C[neg] = (np.cosh(sz) - 1) / (-z[neg])
    S[neg] = (np.sinh(sz) - sz) / sz ** 3
    C[zer] = 0.5 - z[zer] / 24
    S[zer] = 1 / 6 - z[zer] / 120
    return C, S


def propagate_batch(r0, v0, dt, mu=MU, iters=15):
    """Universal-variable Kepler propagation (Laguerre-Conway iterations).
    Returns r, v, ok."""
    R0 = _norm(r0)
    V0 = _norm(v0)
    alpha = 2.0 / R0 - V0 * V0 / mu
    sqmu = np.sqrt(mu)
    vr0 = _dot(r0, v0) / R0
    # initial guess
    chi = np.where(alpha > 0, sqmu * dt * alpha,
                   np.sign(dt) * np.sqrt(-1 / np.minimum(alpha, -1e-30)) *
                   np.log(np.maximum((-2 * mu * alpha * dt) /
                          (_dot(r0, v0) + np.sign(dt) * np.sqrt(-mu / np.minimum(alpha, -1e-30)) * (1 - R0 * alpha)), 1e-30)))
    chi = np.where(np.isfinite(chi), chi, sqmu * dt / R0)
    n = 5.0
    with np.errstate(all="ignore"):
        for _ in range(iters):
            z = alpha * chi * chi
            C, S = _stumpff(z)
            F = R0 * vr0 / sqmu * chi * chi * C + (1 - alpha * R0) * chi ** 3 * S + R0 * chi - sqmu * dt
            dF = R0 * vr0 / sqmu * chi * (1 - z * S) + (1 - alpha * R0) * chi * chi * C + R0
            d2F = R0 * vr0 / sqmu * (1 - z * C) + (1 - alpha * R0) * chi * (1 - z * S)
            disc = np.abs((n - 1) ** 2 * dF * dF - n * (n - 1) * F * d2F)
            den = dF + np.sign(dF) * np.sqrt(disc)
            step = n * F / den
            chi = chi - np.where(np.isfinite(step), step, 0.0)
        z = alpha * chi * chi
        C, S = _stumpff(z)
        f = 1 - chi * chi / R0 * C
        g = dt - chi ** 3 / sqmu * S
        r = f[:, None] * r0 + g[:, None] * v0
        R = _norm(r)
        fd = sqmu / (R * R0) * (alpha * chi ** 3 * S - chi)
        gd = 1 - chi * chi / R * C
        v = fd[:, None] * r0 + gd[:, None] * v0
    ok = np.isfinite(r).all(1) & np.isfinite(v).all(1) & (np.abs(F) < 1e-6 * sqmu * np.maximum(np.abs(dt), 1.0))
    return r, v, ok


# ----------------------------------------------------------------- Lambert (r1 per row)
def lambert_rows(r1, r2, tof, mu=MU, iters=12):
    """Zero-revolution Lambert with one (r1, r2, tof) per row."""
    from .lambert_vec import _x2tof
    c = r2 - r1
    cn = _norm(c)
    R1, R2 = _norm(r1), _norm(r2)
    s = (cn + R1 + R2) / 2.0
    ir1 = r1 / R1[:, None]
    ir2 = r2 / R2[:, None]
    ih = np.cross(ir1, ir2)
    ih /= _norm(ih)[:, None]
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
            x = np.maximum(x, -0.999999)
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
    ok = np.isfinite(v1).all(1) & np.isfinite(v2).all(1) & (resid < 1e-8)
    return v1, v2, ok


# ----------------------------------------------------------------- official safety test
def _anomaly(r, v, mu, a, e):
    """Eccentric / hyperbolic anomaly as returned by pykep.ic2par."""
    h = np.cross(r, v)
    R = _norm(r)
    ev = np.cross(v, h) / mu - r / R[:, None]
    with np.errstate(all="ignore"):
        ni = np.arccos(np.clip(_dot(ev, r) / (e * R), -1.0, 1.0))
        ni = np.where(_dot(r, v) < 0, 2 * np.pi - ni, ni)
        Eell = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(ni / 2))
        arg = np.sqrt((e - 1) / (e + 1)) * np.tan(ni / 2)
        Ehyp = np.arctanh(np.where(np.abs(arg) < 1, arg, np.nan))
    return np.where(e < 1, Eell, Ehyp)


def violation(r0, v0, r1, v1, dt_days, mu=MU):
    """Vectorised copy of trappist_tour._check_distance: returns the value that
    would be max()-ed into the inequality constraint (or -inf if not passed)."""
    R0 = _norm(r0)
    en = _dot(v0, v0) / 2 - mu / R0
    h = np.cross(r0, v0)
    hn2 = _dot(h, h)
    with np.errstate(all="ignore"):
        e = np.sqrt(np.maximum(1 + 2 * en * hn2 / mu ** 2, 0.0))
        a = hn2 / mu / (1 - e * e)
        rp = np.where(e <= 1, a * (1 - e), -a * (1 - e))
        period = 2 * np.pi * np.sqrt(np.where(a > 0, a ** 3 / mu, np.nan)) / DAY2SEC
        E0 = _anomaly(r0, v0, mu, a, e)
        E = _anomaly(r1, v1, mu, a, e)
        passed = np.where(dt_days > period, True,
                          np.where(E0 > 0, (0 < E) & (E < E0), (E > 0) | (dt_days > period / 2)))
    return np.where(passed, SAFE_RADIUS_TRAPPIST - rp, -np.inf)


# ----------------------------------------------------------------- one leg
def leg_batch(r_i, v_pl, v_in, mu_i, eph_j, t_i, beta, rp, eta, T):
    """Evaluate leg i -> j for arrays of (beta, rp, eta, T).

    eph_j: function t_array -> (r (n,3), v (n,3)) of planet j.
    Returns dsm (n,), v_arr (n,3), feasible (n,)."""
    n = len(T)
    v_out = fb_prop_batch(v_in, v_pl, rp, beta, mu_i)
    r0 = np.broadcast_to(r_i, (n, 3))
    t1 = eta * T
    r_d, v_d, ok1 = propagate_batch(r0, v_out, t1 * DAY2SEC)
    r_j, _ = eph_j(t_i + T)
    tau = (1 - eta) * T
    v1, v2, ok2 = lambert_rows(r_d, r_j, tau * DAY2SEC)
    dsm = _norm(v1 - v_d)
    viol = np.maximum(violation(r0, v_out, r_d, v_d, t1), violation(r_d, v1, r_j, v2, tau))
    feas = ok1 & ok2 & np.isfinite(dsm) & ~(viol > 0)
    return np.where(feas, dsm, np.inf), v2, feas
