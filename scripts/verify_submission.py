"""Re-evaluate a submission file and print its score.

    python scripts/verify_submission.py results/final/submission.json
    python scripts/verify_submission.py results/final/submission.json --official

--official uses the untouched ESA evaluation code (needs pykep + pygmo, e.g.
`conda install -c conda-forge pykep pygmo`) and also compares it with this
project's re-implementation, solution by solution.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from trappist.hv import REF, hypervolume
from trappist.problem import udp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--official", action="store_true")
    a = ap.parse_args()
    with open(a.submission) as fh:
        sub = json.load(fh)
    X = sub[0]["decisionVector"]
    assert len(X) <= 100, "at most 100 decision vectors are allowed"
    assert sub[0]["problem"] == "trappist-tour" and sub[0]["challenge"] == "spoc-trappist-tour"

    ours = [udp.fitness(x) for x in X]
    feasible = [f for f in ours if f[2] == 0 and f[3] <= 0]
    print(f"{len(X)} vectors, {len(feasible)} feasible (this project's evaluator)")
    pts = [f[:2] for f in feasible]
    hv = hypervolume(pts, REF)
    print(f"hypervolume = {hv:,.3f}   ->   Optimize score = {-hv:,.3f}")

    if a.official:
        sys.path.insert(0, os.path.join(ROOT, "official"))
        import trappist_tour_official as off       # needs pykep / pygmo
        fo = [off.udp.fitness(x) for x in X]
        d_dv = max(abs(f[0] - g[0]) for f, g in zip(fo, ours))
        d_t = max(abs(f[1] - g[1]) for f, g in zip(fo, ours))
        same_feas = all((f[2] == 0 and f[3] <= 0) == (g[2] == 0 and g[3] <= 0) for f, g in zip(fo, ours))
        print(f"official vs ours: max |dDV| = {d_dv:.3e} m/s, max |dT| = {d_t:.3e} d, "
              f"same feasibility: {same_feas}")
        ok = [f for f in fo if f[2] == 0 and f[3] <= 0]
        print(f"official score = {off.combine_scores([list(f[:2]) for f in ok]):,.3f}")


if __name__ == "__main__":
    main()
