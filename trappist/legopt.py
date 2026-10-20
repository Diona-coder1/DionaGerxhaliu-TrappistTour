"""
Optimisation of single legs i -> j in the exact official encoding
(variables beta, rp/rP, eta, T) with a vectorised Differential Evolution
(DE/current-to-best/1/bin; Storn & Price 1997).

A node of the beam search needs the best leg towards every unvisited planet j
and for several leg-time windows [T_lo, T_hi] (to obtain DV-vs-time
trade-offs). All these (j, window) "tasks" are optimised *simultaneously*:
the populations of all tasks are stacked into one matrix and evaluated with a
single call of vec.leg_batch per generation, which removes the Python/NumPy
overhead that would otherwise dominate.

Cost per node: G generations x (tasks * NP) leg evaluations.
"""
import math

import numpy as np

from .mga import eph_batch
from .problem import PLANETS
from .vec import leg_batch

RP_LO, RP_HI = math.log(1.1), math.log(100.0)
ETA_LO, ETA_HI = 1e-3, 1 - 1e-3

DEFAULT_WINDOWS = ((5, 30), (30, 60), (60, 90), (90, 130), (130, 180), (180, 250), (250, 340))


def _decode(P, T_lo, T_hi):
    beta = P[:, 0] * 2 * np.pi
    rp = np.exp(RP_LO + P[:, 1] * (RP_HI - RP_LO))
    eta = ETA_LO + P[:, 2] * (ETA_HI - ETA_LO)
    T = T_lo + P[:, 3] * (T_hi - T_lo)
    return beta, rp, eta, T


def optimise_legs(state, tasks, NP=40, G=60, rng=None, F=0.6, CR=0.9, seeds=None):
    """state = (i, t_i, r_i, v_pl, v_in); tasks = [(j, T_lo, T_hi), ...].

    seeds: optional dict task_index -> list of 4-vectors in [0,1]^4 inserted in
    the initial population (e.g. ballistic solutions from the cheap model).
    Returns, per task, (dsm, beta, rp, eta, T, v_arr)."""
    rng = rng or np.random.default_rng()
    i, t_i, r_i, v_pl, v_in = state
    mu_i = PLANETS[i].mu_self
    K = len(tasks)
    task_of_row = np.repeat(np.arange(K), NP)
    J = np.array([t[0] for t in tasks])[task_of_row]
    T_lo = np.array([t[1] for t in tasks], dtype=float)[task_of_row]
    T_hi = np.array([t[2] for t in tasks], dtype=float)[task_of_row]
    groups = [(j, np.nonzero(J == j)[0]) for j in np.unique(J)]

    def eph_rows(t):
        r = np.empty((len(t), 3))
        v = np.empty((len(t), 3))
        for j, rows in groups:
            r[rows], v[rows] = eph_batch(int(j), t[rows])
        return r, v

    def evaluate(P):
        beta, rp, eta, T = _decode(P, T_lo, T_hi)
        dsm, v2, ok = leg_batch(r_i, v_pl, v_in, mu_i, eph_rows, t_i, beta, rp, eta, T)
        return dsm, v2

    pop = rng.random((K * NP, 4))
    if seeds:
        for k, lst in seeds.items():
            for m, s in enumerate(lst[:NP // 2]):
                pop[k * NP + m] = np.clip(s, 0, 1)
    fit, _ = evaluate(pop)
    base = (np.arange(K) * NP)[task_of_row]
    for _ in range(G):
        # index of the best individual of each task
        best_idx = base + np.argmin(fit.reshape(K, NP), axis=1)[task_of_row]
        a, b = (base + rng.integers(0, NP, K * NP) for _ in range(2))
        mutant = pop + F * (pop[best_idx] - pop) + F * (pop[a] - pop[b])
        mutant = np.where(mutant < 0, rng.random(mutant.shape) * pop, mutant)
        mutant = np.where(mutant > 1, 1 - rng.random(mutant.shape) * (1 - pop), mutant)
        cross = rng.random(pop.shape) < CR
        cross[np.arange(K * NP), rng.integers(0, 4, K * NP)] = True
        trial = np.where(cross, mutant, pop)
        tf, _ = evaluate(trial)
        better = tf <= fit
        pop[better], fit[better] = trial[better], tf[better]
    best_rows = np.arange(K) * NP + np.argmin(fit.reshape(K, NP), axis=1)
    P = pop[best_rows]
    beta, rp, eta, T = _decode(P, T_lo[best_rows], T_hi[best_rows])
    dsm, v2 = evaluate_rows(state, tasks, P)
    return [(float(dsm[k]), float(beta[k]), float(rp[k]), float(eta[k]), float(T[k]), v2[k])
            for k in range(K)]


def evaluate_rows(state, tasks, P):
    """Evaluate one parameter vector per task (helper for the final answer)."""
    i, t_i, r_i, v_pl, v_in = state
    J = np.array([t[0] for t in tasks])
    T_lo = np.array([t[1] for t in tasks], dtype=float)
    T_hi = np.array([t[2] for t in tasks], dtype=float)

    def eph_rows(t):
        r = np.empty((len(t), 3))
        v = np.empty((len(t), 3))
        for k, j in enumerate(J):
            rr, vv = eph_batch(int(j), t[k:k + 1])
            r[k], v[k] = rr[0], vv[0]
        return r, v
    beta, rp, eta, T = _decode(P, T_lo, T_hi)
    dsm, v2, ok = leg_batch(r_i, v_pl, v_in, PLANETS[i].mu_self, eph_rows, t_i, beta, rp, eta, T)
    return dsm, v2


def encode(beta, rp, eta, T, T_lo, T_hi):
    """Inverse of _decode (to seed the DE with known solutions)."""
    return np.array([beta / (2 * np.pi) % 1.0,
                     (math.log(rp) - RP_LO) / (RP_HI - RP_LO),
                     (eta - ETA_LO) / (ETA_HI - ETA_LO),
                     (T - T_lo) / (T_hi - T_lo)])
