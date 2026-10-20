"""Figures and summary table of the report (reads results/runs/*.pkl and results/final).

usage: python scripts/make_figures.py
"""
import glob
import math
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.astro import DAY2SEC, lambert
from trappist.beam import best_T0, entry_eval
from trappist.hv import REF, hypervolume, nondominated
from trappist.mga import MU, MU_PL, eph_batch, flyby_cost
from trappist.lambert_vec import lambert_batch
from trappist.plotting import LEADERBOARD, plot_fronts, plot_trajectory
from trappist.problem import PLANETS, T_START_MJD2000

R = "results/runs"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def load(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def pts_of(d):
    return np.array([f[:2] for _, f in d["solutions"]]) if d and d["solutions"] else np.zeros((0, 2))


def summary_row(label, ds):
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    hvs = [hypervolume(pts_of(d)) for d in ds]
    secs = [d["seconds"] for d in ds]
    nfeas = [d.get("n_feasible", len(d["solutions"])) for d in ds]
    best = [min((f[0] for _, f in d["solutions"]), default=float("nan")) for d in ds]
    fast = [min((f[1] for _, f in d["solutions"] if f[0] < REF[0]), default=float("nan")) for d in ds]
    return (label, np.mean(hvs), np.std(hvs), np.nanmin(best), np.nanmin(fast), np.mean(nfeas), np.mean(secs), len(ds))


def main():
    rows = []
    rnd = [load(f"random_s{s}.pkl") for s in range(3)]
    nsg = [load(f"nsga2_s{s}.pkl") for s in range(3)]
    rows.append(summary_row("A1 Kërkim i rastësishëm (100k vlerësime, 3 farë)", rnd))
    rows.append(summary_row("A2 NSGA-II (100k vlerësime, 3 farë)", nsg))
    rows.append(summary_row("A3 Greedy (B = 1)", [load("greedy.pkl")]))
    for w in (10, 30, 50, 75, 150, 300):
        name = "beam_main.pkl" if w == 150 else f"beam_w{w}.pkl"
        rows.append(summary_row(f"A4 Beam search, B = {w}", [load(name)]))
    rows.append(summary_row("A4 vetëm Lambert-screening, B = 50", [load("abl_screen_only.pkl")]))
    rows.append(summary_row("A4 vetëm DE, B = 50", [load("abl_de_only.pkl")]))
    rows.append(summary_row("A4 B = 150, T0 i zhvendosur (−40, −80 d)", [load("beam_offsets.pkl")]))
    rows.append(summary_row("A4 B = 300, farë 3, DE G = 90, T0 (0, −20 d)", [load("beam_w300_s3.pkl")]))
    rows = [r for r in rows if r is not None]

    final = None
    if os.path.exists("results/final/final.pkl"):
        with open("results/final/final.pkl", "rb") as fh:
            final = pickle.load(fh)

    with open("results/summary.md", "w") as fh:
        fh.write("| Algoritmi | HV | ΔV min [m/s] | T min* [d] | # të lejuara | Koha [s] | Ekz. |\n")
        fh.write("|:----------------------------------------------|---------:|------:|------:|------:|-----:|---:|\n")
        fmt = lambda v, d=1: "–" if not np.isfinite(v) else f"{v:,.{d}f}"
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]:,.0f} | {fmt(r[3])} | {fmt(r[4])} | {r[5]:,.0f} | {r[6]:,.0f} | {r[7]} |\n")
        if final:
            P = np.array([f[:2] for _, f in final["final"]])
            fh.write(f"| **Pipeline i plotë A4+A5+A6 ({len(P)} zgjidhje)** | **{final['hv']:,.0f}** | "
                     f"{P[:, 0].min():,.1f} | {P[P[:, 0] < REF[0], 1].min():,.1f} | {len(P)} | – | 1 |\n")
    print(open("results/summary.md").read())

    # ---- Fig 1: fronts
    series = []
    if any(d for d in nsg):
        series.append(("A2 NSGA-II (seed 0)", pts_of(nsg[0]) if nsg[0] else [], dict(color="tab:gray", lw=1)))
    g = load("greedy.pkl")
    if g:
        series.append(("A3 Greedy", pts_of(g), dict(color="tab:orange", lw=1, marker="s")))
    for w, c in ((10, "tab:olive"), (150, "tab:blue")):
        d = load("beam_main.pkl" if w == 150 else f"beam_w{w}.pkl")
        if d:
            series.append((f"A4 Beam B={w}", pts_of(d), dict(color=c, lw=1)))
    if final:
        series.append((f"Pipeline final ({len(final['final'])} zgjidhje)", [f[:2] for _, f in final["final"]], dict(color="tab:red", lw=1.6)))
    plot_fronts(series, os.path.join(FIG, "fronti_pareto.png"))

    # ---- Fig 2: width scaling
    ws, hv, sec = [], [], []
    for w in (1, 10, 30, 50, 75, 150, 300):
        name = {1: "greedy.pkl", 150: "beam_main.pkl"}.get(w, f"beam_w{w}.pkl")
        d = load(name)
        if d:
            ws.append(w)
            hv.append(hypervolume(pts_of(d)) / 1e6)
            sec.append(d["seconds"] / 60)
    if ws:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ws, hv, "o-")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Gjerësia e beam-it B")
        ax[0].set_ylabel("Hipervëllimi [×10⁶]")
        ax[0].set_title("Cilësia kundrejt gjerësisë")
        ax[0].grid(alpha=.3)
        ax[1].plot(ws, sec, "o-", color="tab:green")
        if len(ws) > 2:
            k = np.polyfit(ws, sec, 1)
            xx = np.linspace(0, max(ws), 50)
            ax[1].plot(xx, np.polyval(k, xx), "--", color="gray", label=f"përshtatje lineare: {k[0]:.3f}·B + {k[1]:.1f}")
            ax[1].legend()
        ax[1].set_xlabel("Gjerësia e beam-it B")
        ax[1].set_ylabel("Koha [min] (2 bërthama)")
        ax[1].set_title("Koha e ekzekutimit kundrejt gjerësisë")
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "shkallezimi_beam.png"), dpi=140)
        plt.close(fig)

    # ---- Fig 3: entry trade-off
    p, u, v = 1, 0.6, 0.5
    T0s = best_T0(p, u, v)
    TT = np.linspace(T0s - 250, T0s + 150, 200)
    dv = [entry_eval(p, u, v, t)[0] for t in TT]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(TT, dv)
    ax.axvline(T0s, color="gray", ls=":")
    ax.set_xlabel("Koha e hyrjes T0 [ditë]")
    ax.set_ylabel("ΔV_init [m/s]")
    ax.set_title(f"Hapi i hyrjes: ΔV_init = 0 për T0* ≈ {T0s:.0f} d")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "hyrja_T0.png"), dpi=140)
    plt.close(fig)

    # ---- Fig 4: V-shaped screening cost (why 1-D refinement is needed)
    if final:
        x = min(final["final"], key=lambda s: s[1][0])[0]
        seq = [int(s) for s in x[27:]]
        # state at the first planet
        _, _, v_in = entry_eval(seq[0], x[0], x[1], x[2])
        t = T_START_MJD2000 + x[2]
        r_i, v_pl = (np.array(a) for a in PLANETS[seq[0]].eph(t))
        T_star = x[6]
        grid = np.linspace(T_star - 3, T_star + 3, 3001)
        r_j, _ = eph_batch(seq[1], t + grid)
        v1, _, _ = lambert_batch(r_i, r_j, grid * DAY2SEC, MU)
        c, _ = flyby_cost(v_in - v_pl, v1 - v_pl, MU_PL[seq[0]])
        coarse = grid[37::100]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(grid, c, lw=1, label="kosto e saktë (rrjet i dendur)")
        ax.plot(coarse, c[37::100], "o", ms=4, label="rrjet me hap 0.1 d")
        ax.set_yscale("log")
        ax.set_xlabel("Kohëzgjatja e hapit T [ditë]")
        ax.set_ylabel("ΔV i nevojshëm në planet [m/s]")
        ax.set_title("Kostoja e një hapi ka formë 'V' rreth zgjidhjeve balistike")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "kosto_V.png"), dpi=140)
        plt.close(fig)

        # ---- Fig 5: trajectories
        best = min(final["final"], key=lambda s: s[1][0])
        fast = min(final["final"], key=lambda s: s[1][1])
        seqs = lambda xx: "-".join("bcdefgh"[int(s)] for s in xx[27:])
        plot_trajectory(best[0], os.path.join(FIG, "trajektorja_dv_min.png"),
                        f"ΔV = {best[1][0]:.1f} m/s, T = {best[1][1]:.1f} d, sekuenca {seqs(best[0])}")
        plot_trajectory(fast[0], os.path.join(FIG, "trajektorja_e_shpejte.png"),
                        f"ΔV = {fast[1][0]:.1f} m/s, T = {fast[1][1]:.1f} d, sekuenca {seqs(fast[0])}")

        # ---- Fig 6: hypervolume of the final set with leaderboard
        P = np.array(sorted([f[:2] for _, f in final["final"]]))
        fig, ax = plt.subplots(figsize=(7.5, 5))
        xs = np.concatenate([P[:, 0], [REF[0]]])
        for k in range(len(P)):
            ax.add_patch(plt.Rectangle((P[k, 0], P[k, 1]), xs[k + 1] - P[k, 0], REF[1] - P[k, 1],
                                       color="tab:red", alpha=0.25, lw=0))
        ax.scatter(P[:, 0], P[:, 1], s=8, color="tab:red", zorder=3)
        ax.set_xlim(0, REF[0])
        ax.set_ylim(min(P[:, 1]) - 100, REF[1])
        ax.set_xlabel("ΔV [m/s]")
        ax.set_ylabel("Koha [ditë]")
        ax.set_title(f"Hipervëllimi i {len(P)} zgjidhjeve: {final['hv'] / 1e6:.3f}×10⁶ (pika ref. [2500, 4000])")
        txt = "\n".join(f"{n}: {s / 1e6:.3f}×10⁶" for n, s in LEADERBOARD)
        ax.text(0.98, 0.97, "Renditja zyrtare:\n" + txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(facecolor="white", alpha=0.8))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "hipervellimi_final.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
