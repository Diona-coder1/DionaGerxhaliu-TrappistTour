"""
The Trappist Tour problem (ESA SpOC 2022) - faithful port of the official
evaluation code (official/trappist_tour_official.py) on top of trappist/astro.py.

The logic of `fitness` is kept line-by-line identical to the official code,
including its quirks, because the scoring server uses that code:
  * the periapsis radius of a *hyperbolic* arc is computed as -a(1-e) < 0, so
    every hyperbolic arc that passes periapsis violates the star-distance
    constraint;
  * the orbital period of a hyperbolic arc is NaN (numpy semantics), so the
    "dt > period" test is False for it.

Decision vector x (34 numbers):
    x[0:3]   = [u, v, T0]                     entry leg
    x[3:27]  = 6 x [beta, rp/rP, eta, T]      planet-to-planet legs
    x[27:34] = [s0..s6]                       visiting order (permutation of 0..6)
Objectives: total DV [m/s], total time [days]   (both minimised)
Constraints: eq = len(set(seq)) - 7 == 0 ;  ineq <= 0 (safe distance to star)
"""
import math

from .astro import (AU, DAY2SEC, DEG2RAD, EARTH_RADIUS, Planet, fb_prop,
                    ic2par, lambert, propagate_lagrangian)

# ------------------------------------------------------------ constants
G = 6.67430E-11
SM = 1.989E30
EM = 5.9722E24
MS = 8.98266512E-2 * SM
MU_TRAPPIST = G * MS
RADIUS_TRAPPIST = 83484000.0
SAFE_RADIUS_TRAPPIST = 10 * RADIUS_TRAPPIST
DV_PENALTY = 1e10
T_START_MJD2000 = 7000.0          # epoch_from_iso_string("20190302T000000")

MASSES = [m * EM for m in (1.36004499e+00, 1.29688971e+00, 3.85884170e-01, 6.88627613e-01,
                           1.03510927e+00, 1.31679654e+00, 3.20988718e-01)]
_EL = [
    [0.01154035, 0.0158073, 0.02226718, 0.02926548, 0.0385054, 0.04683828, 0.06191385],
    [0.00455247, 0.00107698, 0.00624362, 0.00577129, 0.00861154, 0.00400202, 0.00378361],
    [1.1, 1.3, 0.5, -0.3, 0.01, 1.2, 2.3],
    [238.35940145, 221.02196847, 90.26523049, 164.16001523, 171.42668515, 166.63349296, 282.808809],
    [126.22225948, 80.78036351, 147.44525146, 315.09376191, 179.82855007, 25.31242216, 176.19682451],
    [33.40998384, -89.79506054, -28.38482427, 45.00185452, -145.16404128, 12.59883066, -12.12072197]]
ELEMENTS = []
for k in range(7):
    a, e, i, W, w, M = (_EL[j][k] for j in range(6))
    ELEMENTS.append((a * AU, e, i * DEG2RAD, W * DEG2RAD, w * DEG2RAD, M * DEG2RAD))
SAFE_RADIUS = [f * EARTH_RADIUS for f in (1.016, 1.197, 1.978, 1.020, 1.145, 1.229, 1.875)]
NAMES = ["b", "c", "d", "e", "f", "g", "h"]
PLANETS = [Planet(T_START_MJD2000, ELEMENTS[k], MU_TRAPPIST, G * MASSES[k], EARTH_RADIUS,
                  SAFE_RADIUS[k], "planet_" + NAMES[k]) for k in range(7)]

REF_POINT = (2500.0, 4000.0)
N_DIM = 34


def _hyperbolic_safe_pow(x, p):
    """numpy-like power: negative base with fractional exponent -> NaN."""
    return x ** p if x >= 0 else float("nan")


class TrappistTour:
    """Same interface as the official pygmo UDP."""

    def __init__(self):
        self.n_legs = len(PLANETS)
        self.tof = [[5.0, 2000.0]] * len(PLANETS)
        self.R_START = 10 * AU
        self.V_START = 1e4
        self.common_mu = MU_TRAPPIST

    # ---------------------------------------------------- pygmo interface
    def get_nobj(self):
        return 2

    def get_nix(self):
        return 7

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1

    def get_bounds(self):
        lb = [0.0, 0.25, self.tof[0][0]]
        ub = [1.0, 0.75, self.tof[0][1]]
        for lower_tof, upper_tof in self.tof[1:]:
            lb += [0, 1.1, 1e-3, lower_tof]
            ub += [2 * math.pi, 100.0, 1.0 - 1e-3, upper_tof]
        lb += [0] * 7
        ub += [6] * 7
        return lb, ub

    # ---------------------------------------------------- constraint helpers
    @staticmethod
    def _periapsis_passed(E0, E, dt, period):
        if dt > period:          # False when period is NaN (hyperbola)
            return True
        if E0 > 0:
            return 0 < E < E0
        return E > 0 or (dt > period / 2)

    def _check_distance(self, r0, v0, dt, safe_radius=SAFE_RADIUS_TRAPPIST, mu=MU_TRAPPIST):
        a, e, _, _, _, E0 = ic2par(r0, v0, mu)
        if e <= 1.0:
            rp = a * (1 - e)
        else:
            rp = -a * (1 - e)    # official quirk: negative for hyperbolas
        try:
            r, v = propagate_lagrangian(r0, v0, dt * DAY2SEC, mu)
        except RuntimeError:
            return (True, 10e16)
        period = 2 * math.pi * _hyperbolic_safe_pow(a ** 3 / mu, .5) / DAY2SEC
        _, _, _, _, _, E = ic2par(r, v, mu)
        return (self._periapsis_passed(E0, E, dt, period), safe_radius - rp)

    # ---------------------------------------------------- fitness
    def fitness(self, x, details=False):
        """Returns (DV, T, eq_constraint, ineq_constraint).

        With details=True also returns a dict describing every leg."""
        xc, xi = x[:-7], x[-7:]
        seq = [PLANETS[int(i)] for i in xi]
        eq_constraint = len(set(int(i) for i in xi)) - 7

        u, v = xc[0], xc[1]
        T = xc[2::4]
        betas = xc[3::4]
        rps = xc[4::4]
        etas = xc[5::4]

        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1) - math.pi / 2
        r_init = (self.R_START * math.cos(phi) * math.cos(theta),
                  self.R_START * math.cos(phi) * math.sin(theta),
                  self.R_START * math.sin(phi))

        n = self.n_legs
        r_P, v_P, t_P = [None] * n, [None] * n, [None] * n
        DV = [0.0] * (n - 1)
        iq_constraint = -10e16

        cum = 0.0
        for i in range(n):
            cum += T[i]
            t_P[i] = T_START_MJD2000 + cum
            r_P[i], v_P[i] = seq[i].eph(t_P[i])

        v_beg_l, v_end_l = lambert(r_init, r_P[0], T[0] * DAY2SEC, self.common_mu)
        DV_init = abs(math.sqrt(v_beg_l[0] ** 2 + v_beg_l[1] ** 2 + v_beg_l[2] ** 2) - self.V_START)
        if math.isnan(DV_init):
            DV_init = DV_PENALTY

        close, d = self._check_distance(r_init, v_beg_l, T[0])
        if close:
            iq_constraint = max(iq_constraint, d)

        for i in range(n - 1):
            v_out = fb_prop(v_end_l, v_P[i], rps[i] * seq[i].radius, betas[i], seq[i].mu_self)
            close, d = self._check_distance(r_P[i], v_out, etas[i] * T[i + 1])
            if close:
                iq_constraint = max(iq_constraint, d)
            try:
                r, v = propagate_lagrangian(r_P[i], v_out, etas[i] * T[i + 1] * DAY2SEC, self.common_mu)
            except RuntimeError:
                # pykep would raise here (uncaught in the official code);
                # we mark the solution as unusable instead of crashing.
                return self._penalty(eq_constraint, T, details)
            dt = (1 - etas[i]) * T[i + 1] * DAY2SEC
            v_beg_l, v_end_l = lambert(r, r_P[i + 1], dt, self.common_mu)
            if math.isnan(v_beg_l[0]):
                return self._penalty(eq_constraint, T, details)
            close, d = self._check_distance(r, v_beg_l, (1 - etas[i]) * T[i + 1])
            if close:
                iq_constraint = max(iq_constraint, d)
            DV[i] += math.sqrt((v_beg_l[0] - v[0]) ** 2 + (v_beg_l[1] - v[1]) ** 2 + (v_beg_l[2] - v[2]) ** 2)

        out = (DV_init + sum(DV), sum(T), eq_constraint, iq_constraint)
        if details:
            return out, {"DV_init": DV_init, "DV_legs": DV, "T": list(T),
                         "seq": [int(i) for i in xi], "t_P": t_P}
        return out

    def _penalty(self, eq_constraint, T, details):
        out = (DV_PENALTY, sum(T), eq_constraint, 10e16)
        return (out, None) if details else out

    def pretty(self, x):
        (dv, t, eq, iq), d = self.fitness(x, details=True)
        print(f"== 1 : starting point -> planet_{NAMES[d['seq'][0]]}")
        print(f"Duration: {d['T'][0]:0.6f}d   DV: {d['DV_init']:0.6f}m/s")
        for i in range(self.n_legs - 1):
            a, b = NAMES[d['seq'][i]], NAMES[d['seq'][i + 1]]
            print(f"== {i + 2} : planet_{a} -> planet_{b}   duration {d['T'][i + 1]:0.6f}d   "
                  f"DSM {d['DV_legs'][i]:0.6f}m/s")
        print(f"Total Delta V: {dv:0.3f}m/s   Total mission time: {t:0.6f}d   "
              f"eq={eq}  ineq={'OK' if iq <= 0 else 'VIOLATED'}")

    @staticmethod
    def example():
        return [0.829814333836995, 0.49439092971262544, 1501.6919241751443,
                1.5873030906083387, 1.512021386175137, 0.23024558713584858,
                255.87994897371408, 4.161192563429589, 8.278275659200725, 0.10265621329161562,
                318.6140876998619, 4.546432641418882, 4.078585717496654, 0.33605586368271645,
                374.6287841767241, 0.9461693914845313, 19.13985248269904, 0.4834964750829982,
                343.8204904698013, 2.1935168149524964, 32.48727508444479, 0.40957309270588427,
                325.1537638293067, 4.743610880353654, 49.95062395827792, 0.3668039400426466,
                350.0377039343523, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def is_feasible(f):
    """f = (DV, T, eq, ineq) -> True if both constraints hold (and DV is finite)."""
    return f[2] == 0 and f[3] <= 0 and math.isfinite(f[0]) and math.isfinite(f[1])


udp = TrappistTour()
