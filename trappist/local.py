"""
Local refinement of complete tours (Algorithm 4 in the report).

For a fixed visiting sequence the problem is a 27-dimensional, bound-
constrained continuous problem, but a badly conditioned one: changing a leg
time by one second changes DV by metres per second unless the fly-by
parameters change *with it*. The good solutions lie in long, thin, curved
valleys. Gradient methods (we tried SLSQP) break down on such scales, so we
use CMA-ES (Hansen & Ostermeier 2001), which learns the shape of the valley in
its covariance matrix.

epsilon-constraint formulation (one point of the front per budget):

    minimise   DV(x) + penalties
    subject to sum(T_i) <= budget,   lb <= x <= ub

`trace_front` performs a *continuation*: starting from a polished tour it
lowers the time budget step by step, re-starting CMA-ES from the previous
optimum, which traces the DV-vs-time trade-off of that tour family.
"""
import math

import numpy as np

from .problem import is_feasible, udp

LB, UB = (np.array(b[:27], dtype=float) for b in udp.get_bounds())
SPAN = UB - LB
T_IDX = np.arange(2, 27, 4)              # positions of T0, T1..T6
PENALTY = 1e4


def _split(x):
    x = list(x)
    return np.array(x[:27], dtype=float), [float(s) for s in x[27:]]


def make_objective(seq, budget):
    """z in [0,1]^27 -> penalised DV (feasible & within budget = plain DV)."""
    def full(z):
        return list(LB + np.clip(z, 0.0, 1.0) * SPAN) + seq

    def f(z):
        x = full(z)
        dv, T, eq, iq = udp.fitness(x)
        val = dv if (iq <= 0 and math.isfinite(dv)) else PENALTY + (min(dv, 1e6) if math.isfinite(dv) else 1e6)
        if T > budget:
            val += PENALTY + 1e3 * (T - budget)
        return val, dv, T, iq <= 0 and math.isfinite(dv)
    return f, full


def cmaes(f, z0, sigma0=1e-4, max_evals=4000, rng=None, tol=1e-12, clip=True):
    """(mu/mu_w, lambda)-CMA-ES with box handling by clipping.
    Returns (best z, best value)."""
    rng = rng or np.random.default_rng()
    n = len(z0)
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m = np.array(z0, dtype=float)
    sigma = sigma0
    pc, ps = np.zeros(n), np.zeros(n)
    B, D, C = np.eye(n), np.ones(n), np.eye(n)
    best_z, best_f = m.copy(), f(m)[0]
    evals, gen = 1, 0
    while evals < max_evals:
        arz = rng.standard_normal((lam, n))
        ary = arz * D @ B.T
        arx = m + sigma * ary
        if clip:
            arx = np.clip(arx, 0.0, 1.0)
        fit = np.array([f(x)[0] for x in arx])
        evals += lam
        order = np.argsort(fit)
        if fit[order[0]] < best_f:
            best_f, best_z = fit[order[0]], arx[order[0]].copy()
        y_w = (w @ ((arx[order[:mu]] - m) / sigma))
        m = m + sigma * y_w
        z_w = (B.T @ y_w) / D
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B @ z_w)
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        Y = (arx[order[:mu]] - (m - sigma * y_w)) / sigma
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (Y.T * w) @ Y)
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        gen += 1
        if gen % 5 == 0:
            C = np.triu(C) + np.triu(C, 1).T
            Dsq, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(Dsq, 1e-30))
        if sigma * D.max() < tol * (1 if clip else 1e8):
            break
    return best_z, best_f


# initial step size per variable (in [0,1]-scaled units): leg times are
# ~10^4 times more sensitive than the other variables
SIGMA_VEC = np.full(27, 3e-5)
SIGMA_VEC[T_IDX] = 2e-8


def polish(x, budget=None, max_evals=4000, sigma0=1.0, rng=None):
    """Minimise DV keeping the total time <= budget (default: current time).
    Returns (x, fitness) of the best feasible point, or None."""
    xc, seq = _split(x)
    if budget is None:
        budget = float(np.sum(xc[T_IDX]))
    z0 = np.clip((xc - LB) / SPAN, 0, 1)
    # move the start inside the budget by scaling the leg times
    T = xc[T_IDX]
    if T.sum() > budget:
        xc2 = xc.copy()
        xc2[T_IDX] = T - (T.sum() - budget) * T / T.sum()
        z0 = np.clip((xc2 - LB) / SPAN, 0, 1)
    f, full = make_objective(seq, budget)
    # run CMA-ES in the scaled space y, z = z0 + SIGMA_VEC * y
    g = lambda y: f(np.clip(z0 + SIGMA_VEC * y, 0, 1))
    y, _ = cmaes(g, np.zeros(27), sigma0=sigma0, max_evals=max_evals, rng=rng, clip=False)
    z = np.clip(z0 + SIGMA_VEC * y, 0, 1)
    best = None
    for c in (full(z), list(x)):
        fit = udp.fitness(c)
        if is_feasible(fit) and fit[1] <= budget + 1e-9:
            if best is None or fit[0] < best[1][0]:
                best = (c, fit)
    return best


def shorten(x, delta, which):
    """Remove `delta` days from leg `which` (0 = entry leg)."""
    xc, seq = _split(x)
    k = T_IDX[which]
    xc[k] = max(LB[k], xc[k] - delta)
    return list(xc) + seq


def trace_front(x, step=10.0, dv_max=2500.0, max_steps=40, max_evals=3000, seed=0):
    """Continuation along decreasing time budgets. At every step all ways of
    taking `step` days off (from any single leg, or spread over all legs) are
    tried and the cheapest result is kept. Returns [(x, fitness)]."""
    rng = np.random.default_rng(seed)
    out = []
    p = polish(x, max_evals=max_evals, rng=rng)
    if p is None:
        return out
    out.append(p)
    cur = p
    for _ in range(max_steps):
        budget = cur[1][1] - step
        best = None
        starts = [shorten(cur[0], step, k) for k in range(7)] + [cur[0]]
        for s in starts:
            q = polish(s, budget=budget, max_evals=max_evals // 4, rng=rng)
            if q is not None and (best is None or q[1][0] < best[1][0]):
                best = q
        if best is None or best[1][0] > dv_max:
            break
        best = polish(best[0], budget=budget, max_evals=max_evals, rng=rng) or best
        out.append(best)
        cur = best
    return out
