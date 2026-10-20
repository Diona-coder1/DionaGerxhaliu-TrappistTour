"""
Problem-specific model used by the tree search: MGA with Lambert arcs and
analytic (optimal) fly-bys.

Idea
----
In the official encoding a leg i -> j is: fly-by at i, coast for eta*T,
deep-space manoeuvre (DSM), Lambert arc to j. If the DSM is placed right after
the fly-by (eta -> 0) the leg is fully described by the Lambert arc
r_i(t_i) -> r_j(t_i + T), and the DSM is simply the velocity the fly-by could
not deliver:

    DSM = | v_dep(Lambert) - v_out(best fly-by) |

The best fly-by keeps |v_rel| and turns it by at most delta_max (rp = 1.1 R)
and at least delta_min (rp = 100 R) towards the desired direction, so the
mismatch has a closed form (law of cosines). This turns every leg into ONE
Lambert solve, cheap enough to score millions of candidate legs.
"""
import math

import numpy as np

from .astro import DAY2SEC, EARTH_RADIUS
from .lambert_vec import lambert_batch
from .problem import MU_TRAPPIST, PLANETS, SAFE_RADIUS_TRAPPIST

MU = MU_TRAPPIST
RP_MIN, RP_MAX = 1.1, 100.0          # bounds of rp/rP in the decision vector
MU_PL = np.array([p.mu_self for p in PLANETS])


# ------------------------------------------------------------- ephemerides
def eph_batch(k, t_mjd2000):
    """Vectorised ephemerides of planet k at many epochs -> r (n,3), v (n,3)."""
    p = PLANETS[k]
    a, e, i, W, w, M0 = p.elements
    t = np.atleast_1d(np.asarray(t_mjd2000, dtype=float))
    M = M0 + p.mean_motion * (t - p.ref_mjd2000) * DAY2SEC
    E = M + e * np.cos(M)
    for _ in range(6):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    b = a * math.sqrt(1 - e * e)
    n = p.mean_motion
    cE, sE = np.cos(E), np.sin(E)
    xp, yp = a * (cE - e), b * sE
    xd, yd = -a * n * sE / (1 - e * cE), b * n * cE / (1 - e * cE)
    cW, sW, cw, sw, ci, si = math.cos(W), math.sin(W), math.cos(w), math.sin(w), math.cos(i), math.sin(i)
    R = np.array([[cW * cw - sW * sw * ci, -cW * sw - sW * cw * ci],
                  [sW * cw + cW * sw * ci, -sW * sw + cW * cw * ci],
                  [sw * si, cw * si]])
    r = np.stack([R[0, 0] * xp + R[0, 1] * yp, R[1, 0] * xp + R[1, 1] * yp, R[2, 0] * xp + R[2, 1] * yp], axis=1)
    v = np.stack([R[0, 0] * xd + R[0, 1] * yd, R[1, 0] * xd + R[1, 1] * yd, R[2, 0] * xd + R[2, 1] * yd], axis=1)
    return r, v


# ------------------------------------------------------------- fly-by
def delta_of_rp(rp_ratio, vrel2, mu):
    """Turning angle of a hyperbolic fly-by with periapsis rp_ratio * R."""
    ecc = 1.0 + rp_ratio * EARTH_RADIUS * vrel2 / mu
    return 2.0 * np.arcsin(1.0 / ecc)


def flyby_cost(vin_rel, vout_rel, mu):
    """Minimum |vout_rel - achievable| for many desired outgoing vectors.

    vin_rel: (3,) incoming relative velocity; vout_rel: (n,3) desired outgoing.
    Returns (cost (n,), turn angle used (n,))."""
    vin = np.linalg.norm(vin_rel)
    vout = np.linalg.norm(vout_rel, axis=1)
    cosa = np.clip(vout_rel @ vin_rel / (vin * vout), -1.0, 1.0)
    alpha = np.arccos(cosa)
    dmax = delta_of_rp(RP_MIN, vin * vin, mu)
    dmin = delta_of_rp(RP_MAX, vin * vin, mu)
    d = np.clip(alpha, dmin, dmax)
    cost = np.sqrt(np.maximum(vin * vin + vout * vout - 2 * vin * vout * np.cos(alpha - d), 0.0))
    return cost, d


def flyby_parameters(vin_rel, vout_rel_desired, v_pla, mu):
    """beta and rp/rP that point the fly-by output as close as possible to
    the desired outgoing relative velocity (same frame as pykep.fb_prop)."""
    vin_rel = np.asarray(vin_rel)
    vn = np.linalg.norm(vin_rel)
    i_hat = vin_rel / vn
    j_hat = np.cross(i_hat, v_pla)
    j_hat /= np.linalg.norm(j_hat)
    k_hat = np.cross(i_hat, j_hat)
    d = np.asarray(vout_rel_desired)
    beta = math.atan2(d @ k_hat, d @ j_hat) % (2 * math.pi)
    alpha = math.acos(max(-1.0, min(1.0, d @ i_hat / np.linalg.norm(d))))
    dmax = float(delta_of_rp(RP_MIN, vn * vn, mu))
    dmin = float(delta_of_rp(RP_MAX, vn * vn, mu))
    delta = min(max(alpha, dmin), dmax)
    ecc = 1.0 / math.sin(delta / 2.0)
    rp = (ecc - 1.0) * mu / (vn * vn) / EARTH_RADIUS
    return beta, min(max(rp, RP_MIN), RP_MAX)


# ------------------------------------------------------------- arc safety
def arc_ok(r1, v1, r2, v2):
    """Vectorised version of the official safe-distance test for a Lambert arc.

    An arc is rejected if it passes the periapsis and either the periapsis is
    closer than 10 star radii, or the arc is hyperbolic (official quirk)."""
    h = np.cross(r1, v1)
    R1 = np.linalg.norm(r1, axis=-1)
    V1 = np.linalg.norm(v1, axis=-1)
    energy = V1 * V1 / 2 - MU / R1
    hn2 = np.sum(h * h, axis=-1)
    e = np.sqrt(np.maximum(1 + 2 * energy * hn2 / MU ** 2, 0.0))
    p = hn2 / MU
    rp = p / (1 + e)
    # true anomalies from radial velocity sign and r = p / (1 + e cos nu)
    def nu(r, v):
        R = np.linalg.norm(r, axis=-1)
        c = np.clip((p / R - 1) / np.maximum(e, 1e-12), -1, 1)
        n = np.arccos(c)
        return np.where(np.sum(r * v, axis=-1) < 0, 2 * np.pi - n, n)
    n1, n2 = nu(r1, v1), nu(r2, v2)
    hyper = e >= 1.0
    passed_ell = n2 < n1
    passed_hyp = (n1 > np.pi) & (n2 < np.pi)
    passed = np.where(hyper, passed_hyp, passed_ell)
    bad = passed & (hyper | (rp < SAFE_RADIUS_TRAPPIST * 1.001))
    return ~bad
