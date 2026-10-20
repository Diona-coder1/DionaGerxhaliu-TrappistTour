"""
Generic (problem-agnostic) baselines, used to show what the problem-specific
algorithms gain.

Algorithm 1 - Random search
    Sample x uniformly in the bounds (sequence = random permutation),
    keep the non-dominated feasible points.            Cost: O(N) evaluations.

Algorithm 2 - NSGA-II (Deb et al., 2002) for the mixed encoding
    * real part (27 genes, scaled to [0,1]): SBX crossover + polynomial mutation
    * permutation part (7 genes): order crossover (OX1) + swap mutation
    * constraints: Deb's constrained-domination (feasible solutions first,
      infeasible ones ranked by constraint violation)
    * selection: binary tournament on (rank, crowding distance)
    Cost per generation: O(N) evaluations + O(M N^2) sorting (M = 2 objectives).
"""
import math
import time

import numpy as np

from .problem import udp

LB, UB = (np.array(b[:27], dtype=float) for b in udp.get_bounds())
SPAN = UB - LB


def decode(z, perm):
    return list(LB + np.clip(z, 0, 1) * SPAN) + [float(s) for s in perm]


def evaluate(z, perm):
    dv, T, eq, iq = udp.fitness(decode(z, perm))
    if not math.isfinite(dv):
        dv = 1e10
    viol = 0.0 if iq <= 0 else 1.0 + iq / 1e9
    return np.array([dv, T]), viol


# ------------------------------------------------------------------ random search
def random_search(n_evals, seed=0, log=None):
    rng = np.random.default_rng(seed)
    pts, xs = [], []
    t0 = time.time()
    for k in range(n_evals):
        z = rng.random(27)
        perm = rng.permutation(7)
        f, viol = evaluate(z, perm)
        if viol == 0:
            pts.append(f)
            xs.append(decode(z, perm))
        if log and (k + 1) % 10000 == 0:
            log(f"  random search {k + 1}/{n_evals}, feasible {len(pts)} ({time.time() - t0:.0f}s)")
    return xs, pts


# ------------------------------------------------------------------ NSGA-II
def fast_nondominated_sort(F, viol):
    """Constrained non-dominated sorting -> rank array (0 = best front)."""
    n = len(F)
    feas = viol == 0
    dom = np.zeros((n, n), dtype=bool)          # dom[i, j]: i dominates j
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom_obj = le & lt
    dom = np.where(feas[:, None] & feas[None, :], dom_obj, False)
    dom |= feas[:, None] & ~feas[None, :]
    both_inf = ~feas[:, None] & ~feas[None, :]
    dom |= both_inf & (viol[:, None] < viol[None, :])
    n_dom = dom.sum(0)
    rank = np.full(n, -1)
    r, current = 0, np.nonzero(n_dom == 0)[0]
    while len(current):
        rank[current] = r
        n_dom = n_dom - dom[current].sum(0)
        n_dom[rank >= 0] = -1
        current = np.nonzero(n_dom == 0)[0]
        r += 1
    return rank


def crowding(F, rank):
    cd = np.zeros(len(F))
    for r in np.unique(rank):
        idx = np.nonzero(rank == r)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf
            continue
        for m in range(F.shape[1]):
            o = idx[np.argsort(F[idx, m])]
            cd[o[0]] = cd[o[-1]] = np.inf
            span = F[o[-1], m] - F[o[0], m]
            if span > 0:
                cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return cd


def sbx(a, b, rng, eta=15.0, p=0.9):
    if rng.random() > p:
        return a.copy(), b.copy()
    u = rng.random(a.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    swap = rng.random(a.shape) < 0.5
    c1, c2 = np.where(swap, c2, c1), np.where(swap, c1, c2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def poly_mut(z, rng, eta=20.0, p=None):
    p = p or 1.0 / len(z)
    m = rng.random(z.shape) < p
    u = rng.random(z.shape)
    d = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    return np.clip(np.where(m, z + d, z), 0, 1)


def ox1(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in child[a:b + 1]]
    child[[k for k in range(n) if child[k] < 0]] = fill
    return child


def swap_mut(perm, rng, p=0.2):
    perm = perm.copy()
    if rng.random() < p:
        a, b = rng.choice(len(perm), 2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    return perm


def nsga2(n_evals, pop_size=100, seed=0, init=None, log=None):
    """Returns (archive_x, archive_f) of all feasible evaluated points."""
    rng = np.random.default_rng(seed)
    Z = rng.random((pop_size, 27))
    P = np.array([rng.permutation(7) for _ in range(pop_size)])
    if init:
        for k, x in enumerate(init[:pop_size // 2]):
            Z[k] = (np.array(x[:27]) - LB) / SPAN
            P[k] = np.array(x[27:], dtype=int)
    ev = [evaluate(Z[k], P[k]) for k in range(pop_size)]
    F = np.array([e[0] for e in ev])
    V = np.array([e[1] for e in ev])
    arch_x, arch_f = [], []

    def archive(z, p, f, v):
        if v == 0:
            arch_x.append(decode(z, p))
            arch_f.append(f)
    for k in range(pop_size):
        archive(Z[k], P[k], F[k], V[k])
    n_done, gen, t0 = pop_size, 0, time.time()
    while n_done < n_evals:
        rank = fast_nondominated_sort(F, V)
        cd = crowding(F, rank)

        def tournament():
            a, b = rng.integers(0, pop_size, 2)
            if rank[a] != rank[b]:
                return a if rank[a] < rank[b] else b
            return a if cd[a] > cd[b] else b
        Zc, Pc = [], []
        while len(Zc) < pop_size:
            a, b = tournament(), tournament()
            c1, c2 = sbx(Z[a], Z[b], rng)
            Zc += [poly_mut(c1, rng), poly_mut(c2, rng)]
            Pc += [swap_mut(ox1(P[a], P[b], rng), rng), swap_mut(ox1(P[b], P[a], rng), rng)]
        Zc, Pc = np.array(Zc[:pop_size]), np.array(Pc[:pop_size])
        ev = [evaluate(Zc[k], Pc[k]) for k in range(pop_size)]
        Fc = np.array([e[0] for e in ev])
        Vc = np.array([e[1] for e in ev])
        for k in range(pop_size):
            archive(Zc[k], Pc[k], Fc[k], Vc[k])
        n_done += pop_size
        # environmental selection (mu + lambda)
        Zall, Pall = np.vstack([Z, Zc]), np.vstack([P, Pc])
        Fall, Vall = np.vstack([F, Fc]), np.concatenate([V, Vc])
        rank = fast_nondominated_sort(Fall, Vall)
        cd = crowding(Fall, rank)
        order = np.lexsort((-cd, rank))[:pop_size]
        Z, P, F, V = Zall[order], Pall[order], Fall[order], Vall[order]
        gen += 1
        if log and gen % 50 == 0:
            log(f"  NSGA-II gen {gen}, evals {n_done}, feasible archive {len(arch_f)} ({time.time() - t0:.0f}s)")
    return arch_x, arch_f
