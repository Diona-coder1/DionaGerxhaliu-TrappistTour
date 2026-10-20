"""
Hypervolume utilities for 2 objectives (both minimised).

* nondominated(points)            O(n log n)
* hypervolume(points, ref)        O(n log n)   (same value as pygmo.hypervolume)
* best_subset(points, k, ref)     O(k n^2)     exact dynamic programming

The competition accepts at most 100 decision vectors, so after optimisation we
must choose the subset of at most k points with the largest hypervolume. In 2-D
this "hypervolume subset selection problem" has optimal substructure:
sort the non-dominated points by f1 (then f2 is decreasing). If the chosen
points are i1 < i2 < ... < ik, the dominated area splits into rectangles
    (f1[i_{m+1}] - f1[i_m]) * (ref2 - f2[i_m]),   f1[i_{k+1}] := ref1
so the contribution of a point depends only on the NEXT chosen point:

    best[c][i] = max_{j > i} ( (f1[j] - f1[i]) * (ref2 - f2[i]) + best[c-1][j] )
    best[1][i] = (ref1 - f1[i]) * (ref2 - f2[i])

(Bringmann, Friedrich & Klitzke, 2014 give an O(k n log n) refinement.)
"""
import numpy as np

REF = (2500.0, 4000.0)


def nondominated(points, ref=REF):
    """Indices of the non-dominated points that strictly dominate `ref`,
    sorted by the first objective."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.array([], dtype=int)
    inside = np.nonzero((pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1]))[0]
    order = inside[np.lexsort((pts[inside, 1], pts[inside, 0]))]
    keep, best_f2 = [], np.inf
    for k in order:
        if pts[k, 1] < best_f2:
            keep.append(k)
            best_f2 = pts[k, 1]
    return np.array(keep, dtype=int)


def hypervolume(points, ref=REF):
    pts = np.asarray(points, dtype=float)
    idx = nondominated(pts, ref)
    if len(idx) == 0:
        return 0.0
    f1, f2 = pts[idx, 0], pts[idx, 1]
    nxt = np.append(f1[1:], ref[0])
    return float(np.sum((nxt - f1) * (ref[1] - f2)))


def best_subset(points, k=100, ref=REF):
    """Indices (into `points`) of the <= k points maximising the hypervolume."""
    pts = np.asarray(points, dtype=float)
    idx = nondominated(pts, ref)
    n = len(idx)
    if n <= k:
        return idx
    f1, f2 = pts[idx, 0], pts[idx, 1]
    h = ref[1] - f2                       # height of each rectangle
    best = (ref[0] - f1) * h              # c = 1
    choice = []                           # choice[c][i] = next index after i
    for _ in range(2, k + 1):
        new = np.full(n, -np.inf)
        nxt = np.full(n, -1)
        for i in range(n - 1):
            cand = (f1[i + 1:] - f1[i]) * h[i] + best[i + 1:]
            j = int(np.argmax(cand))
            new[i], nxt[i] = cand[j], i + 1 + j
        choice.append(nxt)
        best = np.maximum(best, new)      # "at most" c points
        # keep track of whether using c points helped
        choice[-1] = np.where(new >= best, nxt, -1)
    # reconstruct
    i = int(np.argmax(best))
    chosen = [i]
    for c in range(len(choice) - 1, -1, -1):
        j = choice[c][i]
        if j < 0:
            continue
        chosen.append(int(j))
        i = int(j)
    chosen = sorted(set(chosen))
    return idx[chosen]


def _brute_force_check():
    import itertools
    rng = np.random.default_rng(0)
    for _ in range(30):
        pts = rng.uniform(0, 1, size=(9, 2)) * np.array(REF)
        idx = nondominated(pts)
        for k in (1, 2, 3):
            if len(idx) <= k:
                continue
            exact = max(hypervolume(pts[list(c)]) for c in itertools.combinations(idx, k))
            got = hypervolume(pts[best_subset(pts, k)])
            assert abs(exact - got) < 1e-6 * max(1, exact), (exact, got)
    return True


if __name__ == "__main__":
    print("subset DP matches brute force:", _brute_force_check())
