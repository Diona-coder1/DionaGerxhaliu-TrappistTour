"""Vectorised leg evaluation must agree with the scalar fitness function."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.astro import DAY2SEC, propagate_lagrangian
from trappist.mga import eph_batch
from trappist.problem import MU_TRAPPIST, PLANETS, T_START_MJD2000, udp
from trappist.vec import leg_batch, propagate_batch


def test_propagate_batch():
    rng = np.random.default_rng(3)
    r0 = rng.normal(size=(200, 3)) * 2e9
    v0 = rng.normal(size=(200, 3)) * 6e4
    dt = rng.uniform(0.01, 300, 200) * DAY2SEC
    r, v, ok = propagate_batch(r0, v0, dt)
    n = 0
    for k in range(200):
        try:
            rs, vs = propagate_lagrangian(tuple(r0[k]), tuple(v0[k]), dt[k], MU_TRAPPIST)
        except RuntimeError:
            continue
        if ok[k]:
            n += 1
            assert np.linalg.norm(r[k] - rs) / np.linalg.norm(rs) < 1e-7
    assert n > 150


def test_leg_matches_fitness():
    """Re-compute the DSMs of the official example leg by leg."""
    x = udp.example()
    (dv, T, eq, iq), d = udp.fitness(x, details=True)
    from trappist.astro import lambert
    seq = d["seq"]
    # incoming velocity at the first planet = arrival of the entry Lambert arc
    import math
    theta, phi = 2 * math.pi * x[0], math.acos(2 * x[1] - 1) - math.pi / 2
    R = udp.R_START
    r0 = (R * math.cos(phi) * math.cos(theta), R * math.cos(phi) * math.sin(theta), R * math.sin(phi))
    t = T_START_MJD2000 + x[2]
    rP, _ = PLANETS[seq[0]].eph(t)
    _, v_in = lambert(r0, rP, x[2] * DAY2SEC, MU_TRAPPIST)
    v_in = np.array(v_in)
    for k in range(6):
        beta, rp, eta, Tk = x[3 + 4 * k: 7 + 4 * k]
        i, j = seq[k], seq[k + 1]
        r_i, v_pl = PLANETS[i].eph(t)
        dsm, v2, feas = leg_batch(np.array(r_i), np.array(v_pl), v_in, PLANETS[i].mu_self,
                                  lambda tt: eph_batch(j, tt), t,
                                  np.array([beta]), np.array([rp]), np.array([eta]), np.array([Tk]))
        assert abs(dsm[0] - d["DV_legs"][k]) < 1e-3, (k, dsm[0], d["DV_legs"][k])
        v_in = v2[0]
        t += Tk


if __name__ == "__main__":
    test_propagate_batch()
    test_leg_matches_fitness()
    print("OK")
