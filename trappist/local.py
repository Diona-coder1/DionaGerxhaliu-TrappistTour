"""
Local refinement of complete tours (Algorithm 4 in the report).

For a fixed visiting sequence the problem is a smooth, 27-dimensional,
bound-constrained continuous problem. We use the epsilon-constraint method:

    minimise   DV(x)
    subject to sum(T_i) <= budget        (linear)
               lb <= x <= ub

solved with SLSQP (sequential quadratic programming, gradients by finite
differences). Variables are scaled to [0, 1] so that one finite-difference
step means the same thing for u (in [0,1]) and for a leg time (in days).

`trace_front` performs a *continuation*: starting from a polished tour it
lowers the time budget step by step, re-optimising from the previous optimum,
which traces the DV-vs-time trade-off curve of that tour family.
"""
import math

import numpy as np
from scipy.optimize import minimize

from .problem import is_feasible, udp

LB, UB = (np.array(b[:27], dtype=float) for b in udp.get_bounds())
SPAN = UB - LB
T_IDX = np.arange(2, 27, 4)              # positions of T0, T1..T6
PENALTY = 1e4


def _split(x):
    x = list(x)
    return np.array(x[:27], dtype=float), [float(s) for s in x[27:]]


def objective_factory(seq):
    cache = {}

    def full(z):
        xc = LB + np.clip(z, 0.0, 1.0) * SPAN
        return list(xc) + seq

    def f(z):
        key = z.tobytes()
        if key not in cache:
            dv, T, eq, iq = udp.fitness(full(z))
            val = dv if (iq <= 0 and math.isfinite(dv)) else PENALTY + min(dv, 1e6)
            cache[key] = val
            if len(cache) > 5000:
                cache.clear()
        return cache[key]
    return f, full


def polish(x, budget=None, maxiter=80):
    """Minimise DV keeping the total time <= budget (default: current time)."""
    xc, seq = _split(x)
    z0 = np.clip((xc - LB) / SPAN, 0, 1)
    f, full = objective_factory(seq)
    if budget is None:
        budget = float(np.sum(xc[T_IDX]))
    a = np.zeros(27)
    a[T_IDX] = SPAN[T_IDX]
    b = budget - float(np.sum(LB[T_IDX]))
    cons = [{"type": "ineq", "fun": lambda z: b - a @ z, "jac": lambda z: -a}]
    res = minimize(f, z0, method="SLSQP", bounds=[(0, 1)] * 27, constraints=cons,
                   options={"maxiter": maxiter, "ftol": 1e-6, "eps": 1e-7})
    cand = [full(z0), full(res.x)]
    best = None
    for c in cand:
        fit = udp.fitness(c)
        if is_feasible(fit) and fit[1] <= budget + 1e-6:
            if best is None or fit[0] < best[1][0]:
                best = (c, fit)
    return best


def shift_time(x, delta):
    """Scale all leg times so that the total time changes by `delta` days
    (used as the starting point of the next continuation step)."""
    xc, seq = _split(x)
    T = xc[T_IDX]
    xc[T_IDX] = np.clip(T + delta * T / T.sum(), LB[T_IDX], UB[T_IDX])
    return list(xc) + seq


def trace_front(x, step=10.0, dv_max=2500.0, max_steps=60, maxiter=60):
    """Continuation along decreasing time budgets. Returns [(x, fitness)]."""
    out = []
    p = polish(x, maxiter=maxiter)
    if p is None:
        return out
    out.append(p)
    cur = p
    for _ in range(max_steps):
        budget = cur[1][1] - step
        q = polish(cur[0], budget=budget, maxiter=maxiter)
        if q is None or q[1][0] > dv_max:
            break
        out.append(q)
        cur = q
    return out
