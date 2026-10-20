"""Run the hybrid beam search (Algorithm 3) and store all complete tours.

usage: python scripts/run_beam.py --width 150 --root-width 245 --out results/runs/beam_main.pkl
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.beam import beam_search, entry_nodes, node_to_x
from trappist.hv import hypervolume
from trappist.problem import is_feasible, udp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=150)
    ap.add_argument("--root-width", type=int, default=245)
    ap.add_argument("--nu", type=int, default=24)
    ap.add_argument("--nv", type=int, default=7)
    ap.add_argument("--de-G", type=int, default=60)
    ap.add_argument("--no-de", action="store_true")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/runs/beam_main.pkl")
    a = ap.parse_args()

    t0 = time.time()
    roots = entry_nodes(n_u=a.nu, n_v=a.nv)
    print(f"{len(roots)} feasible entry nodes ({time.time() - t0:.1f}s)", flush=True)
    kw = dict(de_G=a.de_G, use_de=not a.no_de, use_screen=not a.no_screen, seed=a.seed)
    with Pool(a.workers) as pool:
        final = beam_search(roots, width=a.width, root_width=a.root_width, expand_kw=kw, pool=pool,
                            log=lambda m: print(m, flush=True))
    sols = []
    for n in final:
        x = node_to_x(n)
        f = udp.fitness(x)
        if is_feasible(f):
            sols.append((x, f))
    pts = [f[:2] for _, f in sols]
    print(f"complete feasible tours: {len(sols)}   HV = {hypervolume(pts):,.0f}   "
          f"time {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"solutions": sols, "args": vars(a), "seconds": time.time() - t0}, fh)


if __name__ == "__main__":
    main()
