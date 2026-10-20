"""Stages C + D of the pipeline: local refinement + continuation, then the
hypervolume-optimal choice of <= 100 solutions and the submission file.

usage: python scripts/run_front.py results/runs/beam_main.pkl [more.pkl ...]
                                   --starts 24 --out results/final
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.hv import REF, best_subset, hypervolume, nondominated
from trappist.local import trace_front
from trappist.problem import is_feasible, udp


def pick_starts(sols, k):
    """Diverse starting tours: best of several weighted sums, one per sequence."""
    chosen, seqs = [], set()
    lams = [0.0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 25, 50]
    rounds = 0
    while len(chosen) < k and rounds < 20:
        for lam in lams:
            order = sorted(range(len(sols)), key=lambda i: sols[i][1][0] + lam * sols[i][1][1])
            for i in order:
                seq = tuple(int(s) for s in sols[i][0][27:])
                key = (seq, rounds)
                if seq in seqs and rounds == 0:
                    continue
                if i in [c for c in chosen]:
                    continue
                chosen.append(i)
                seqs.add(seq)
                break
            if len(chosen) >= k:
                break
        rounds += 1
    return [sols[i] for i in chosen]


def _trace(args):
    x, step, maxiter = args
    return trace_front(x, step=step, maxiter=maxiter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()

    sols = []
    for path in a.inputs:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        sols += d["solutions"]
    print(f"{len(sols)} input solutions, HV before refinement = {hypervolume([f[:2] for _, f in sols]):,.0f}")
    starts = pick_starts(sols, a.starts)
    t0 = time.time()
    with Pool(a.workers) as pool:
        curves = pool.map(_trace, [(x, a.step, a.maxiter) for x, _ in starts])
    archive = list(sols)
    for c in curves:
        archive += c
    archive = [(x, f) for x, f in archive if is_feasible(f)]
    pts = np.array([f[:2] for _, f in archive])
    print(f"after continuation: {len(archive)} solutions, HV = {hypervolume(pts):,.0f} ({time.time() - t0:.0f}s)")

    idx = best_subset(pts, 100, REF)
    final = [archive[i] for i in idx]
    hv = hypervolume([f[:2] for _, f in final])
    print(f"selected {len(final)} solutions, HV = {hv:,.0f}  (score {-hv:,.3f})")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "archive.pkl"), "wb") as fh:
        pickle.dump({"archive": archive, "final": final, "hv": hv}, fh)
    sub = [{"decisionVector": [list(map(float, x)) for x, _ in final],
            "problem": "trappist-tour", "challenge": "spoc-trappist-tour"}]
    with open(os.path.join(a.out, "submission.json"), "w") as fh:
        json.dump(sub, fh)
    with open(os.path.join(a.out, "front.csv"), "w") as fh:
        fh.write("DV_m_s,T_days,sequence\n")
        for x, f in sorted(final, key=lambda s: s[1][0]):
            fh.write(f"{f[0]:.3f},{f[1]:.3f},{'-'.join('bcdefgh'[int(s)] for s in x[27:])}\n")


if __name__ == "__main__":
    main()
