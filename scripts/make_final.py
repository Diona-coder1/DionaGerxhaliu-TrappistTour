"""Final assembly (stages C + D of the pipeline):

  1. union of the complete tours of all beam-search runs
  2. CMA-ES polishing (Algorithm 4) of the best non-dominated layers
  3. exact hypervolume-optimal subset of <= 100 solutions (Algorithm 5)
  4. submission.json in the official Optimize format + front.csv

usage: python scripts/make_final.py results/runs/beam_*.pkl --polish 200 --out results/final
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
from trappist.local import polish
from trappist.problem import is_feasible, udp


def layers(pts, max_layers=3):
    """Indices of the first `max_layers` non-dominated layers."""
    remaining = np.arange(len(pts))
    out = []
    for _ in range(max_layers):
        if len(remaining) == 0:
            break
        nd = remaining[nondominated(pts[remaining])]
        out += list(nd)
        remaining = np.setdiff1d(remaining, nd)
    return out


def _polish(args):
    x, seed, evals = args
    return polish(x, max_evals=evals, rng=np.random.default_rng(seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--polish", type=int, default=200)
    ap.add_argument("--evals", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()

    sols = []
    for path in a.inputs:
        with open(path, "rb") as fh:
            sols += pickle.load(fh)["solutions"]
    sols = [(x, f) for x, f in sols if is_feasible(f)]
    pts = np.array([f[:2] for _, f in sols])
    print(f"{len(sols)} feasible tours from {len(a.inputs)} runs, union HV = {hypervolume(pts):,.0f}", flush=True)

    t0 = time.time()
    idx = layers(pts, 3)[: a.polish]
    with Pool(a.workers) as pool:
        res = pool.map(_polish, [(sols[i][0], k, a.evals) for k, i in enumerate(idx)])
    polished = [r for r in res if r is not None]
    archive = sols + polished
    pts = np.array([f[:2] for _, f in archive])
    print(f"polished {len(polished)} tours in {time.time() - t0:.0f}s, HV = {hypervolume(pts):,.0f}", flush=True)

    sel = best_subset(pts, 100, REF)
    final = [archive[i] for i in sel]
    # re-check every selected vector with a fresh evaluation
    final = [(x, udp.fitness(x)) for x, _ in final]
    assert all(is_feasible(f) for _, f in final)
    hv = hypervolume([f[:2] for _, f in final])
    print(f"selected {len(final)} solutions, HV = {hv:,.3f}  ->  Optimize score {-hv:,.3f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "final.pkl"), "wb") as fh:
        pickle.dump({"final": final, "hv": hv, "union_points": pts}, fh)
    sub = [{"decisionVector": [[float(v) for v in x] for x, _ in final],
            "problem": "trappist-tour", "challenge": "spoc-trappist-tour"}]
    with open(os.path.join(a.out, "submission.json"), "w") as fh:
        json.dump(sub, fh, indent=1)
    with open(os.path.join(a.out, "front.csv"), "w") as fh:
        fh.write("DV_m_s,T_days,T0_days,sequence\n")
        for x, f in sorted(final, key=lambda s: s[1][0]):
            fh.write(f"{f[0]:.4f},{f[1]:.4f},{x[2]:.3f},{'-'.join('bcdefgh'[int(s)] for s in x[27:])}\n")


if __name__ == "__main__":
    main()
