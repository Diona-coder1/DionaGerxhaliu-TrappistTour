"""
Independent validation of trappist/astro.py.

pykep cannot be installed in every environment, so each routine is checked
against an *independent* method: direct numerical integration of the
two-body equations of motion with SciPy (RK45/DOP853, rtol 1e-12).

Run:  python -m pytest -q tests      (or: python tests/test_astro.py)
"""
import math
import os
import random
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.astro import (AU, DAY2SEC, fb_prop, ic2par, lambert, par2ic,
                            propagate_lagrangian)
from trappist.problem import MU_TRAPPIST, PLANETS, udp

MU = MU_TRAPPIST


def integrate(r0, v0, t):
    def rhs(_, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU * r / np.linalg.norm(r) ** 3])
    sol = solve_ivp(rhs, (0, t), np.concatenate([r0, v0]), method="DOP853",
                    rtol=1e-12, atol=1e-6)
    return sol.y[:3, -1], sol.y[3:, -1]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_propagate_lagrangian_vs_integration():
    rng = random.Random(1)
    for _ in range(20):
        r0 = np.array([rng.uniform(-1, 1) for _ in range(3)]) * 0.05 * AU
        vc = math.sqrt(MU / np.linalg.norm(r0))
        v0 = np.array([rng.uniform(-1, 1) for _ in range(3)]) * vc * rng.choice([0.5, 1.0, 1.4])
        t = rng.uniform(0.2, 30) * DAY2SEC
        r1, v1 = propagate_lagrangian(tuple(r0), tuple(v0), t, MU)
        r2, v2 = integrate(r0, v0, t)
        assert rel(r1, r2) < 1e-7 and rel(v1, v2) < 1e-7


def test_lambert_reaches_target():
    rng = random.Random(2)
    for _ in range(50):
        p1, p2 = rng.sample(PLANETS, 2)
        t1 = rng.uniform(7000, 9000)
        tof = rng.uniform(0.5, 300)
        r1, _ = p1.eph(t1)
        r2, _ = p2.eph(t1 + tof)
        v1, v2 = lambert(r1, r2, tof * DAY2SEC, MU)
        rf, vf = propagate_lagrangian(r1, v1, tof * DAY2SEC, MU)
        assert rel(rf, r2) < 1e-8, rel(rf, r2)
        assert rel(vf, v2) < 1e-6


def test_lambert_vs_integration_entry_leg():
    # 10 AU -> planet b, the hyperbolic entry arc of the example
    x = udp.example()
    theta, phi = 2 * math.pi * x[0], math.acos(2 * x[1] - 1) - math.pi / 2
    r0 = 10 * AU * np.array([math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)])
    rP, _ = PLANETS[0].eph(7000 + x[2])
    v1, v2 = lambert(tuple(r0), rP, x[2] * DAY2SEC, MU)
    rf, vf = integrate(r0, np.array(v1), x[2] * DAY2SEC)
    assert rel(rf, rP) < 1e-6
    assert rel(vf, v2) < 1e-6


def test_ic2par_par2ic_roundtrip():
    for p in PLANETS:
        r, v = p.eph(7123.4)
        a, e, i, W, w, E = ic2par(r, v, MU)
        r2, v2 = par2ic((a, e, i, W, w, E), MU)
        assert rel(r2, r) < 1e-10 and rel(v2, v) < 1e-10
        assert abs(a - p.elements[0]) / p.elements[0] < 1e-10


def test_planet_period():
    for p in PLANETS:
        r0, v0 = p.eph(7000.0)
        r1, v1 = p.eph(7000.0 + p.period_days)
        assert rel(r1, r0) < 1e-9


def test_flyby_conserves_relative_speed():
    v_pla = (70e3, 10e3, 1e3)
    v_in = (90e3, -5e3, 2e3)
    mu = PLANETS[0].mu_self
    vout = fb_prop(v_in, v_pla, 1.5 * 6378137.0, 1.0, mu)
    rin = np.subtract(v_in, v_pla)
    rout = np.subtract(vout, v_pla)
    assert abs(np.linalg.norm(rin) - np.linalg.norm(rout)) < 1e-6
    # turning angle = 2 asin(1/e)
    ecc = 1 + 1.5 * 6378137.0 / mu * np.dot(rin, rin)
    ang = math.acos(np.dot(rin, rout) / np.dot(rin, rin))
    assert abs(ang - 2 * math.asin(1 / ecc)) < 1e-9


def test_example_fitness():
    dv, t, eq, iq = udp.fitness(udp.example())
    assert abs(t - 3469.8267032589047) < 1e-9
    assert eq == 0 and iq <= 0
    assert abs(dv - 2378.99) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
