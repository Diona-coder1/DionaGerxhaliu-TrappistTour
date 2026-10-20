"""Random search and NSGA-II baselines (Algorithms 1 and 2).

usage: python scripts/run_baselines.py --algo nsga2 --evals 100000 --seed 0
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.baselines import nsga2, random_search
from trappist.hv import hypervolume
from trappist.problem import udp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["random", "nsga2"], required=True)
    ap.add_argument("--evals", type=int, default=100000)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/runs/{a.algo}_s{a.seed}.pkl"
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    if a.algo == "random":
        xs, pts = random_search(a.evals, seed=a.seed, log=log)
    else:
        xs, pts = nsga2(a.evals, pop_size=a.pop, seed=a.seed, log=log)
    sols = [(x, udp.fitness(x)) for x, _ in zip(xs, pts)]
    hv = hypervolume([f[:2] for _, f in sols]) if sols else 0.0
    best_dv = min((f[0] for _, f in sols), default=float("nan"))
    print(f"{a.algo} seed {a.seed}: {len(sols)} feasible, best DV {best_dv:.1f}, HV = {hv:,.0f}, "
          f"time {time.time() - t0:.0f}s", flush=True)
    with open(out, "wb") as fh:
        pickle.dump({"solutions": sols, "args": vars(a), "seconds": time.time() - t0, "hv": hv}, fh)


if __name__ == "__main__":
    main()
