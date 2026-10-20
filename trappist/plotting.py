"""Figures: Pareto fronts and trajectory plots (no pykep needed)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .astro import AU, DAY2SEC, fb_prop, lambert, propagate_lagrangian
from .hv import REF, nondominated
from .problem import MU_TRAPPIST, NAMES, PLANETS, T_START_MJD2000, udp

COLORS = ["coral", "seagreen", "purple", "steelblue", "firebrick", "gray", "olive"]
LEADERBOARD = [("1. Space Coders (2022)", 5870436.818), ("3. Team HRI", 5601395.897),
               ("9. Alien Colony Opt.", 4161632.839)]


def _arc(r0, v0, dt_s, n=300):
    pts = [r0]
    for k in range(1, n + 1):
        r, _ = propagate_lagrangian(r0, v0, dt_s * k / n, MU_TRAPPIST)
        pts.append(r)
    return np.array(pts) / AU


def trajectory_arcs(x):
    """List of (kind, points) for every arc of decision vector x."""
    seq = [int(s) for s in x[27:]]
    u, v, T0 = x[0], x[1], x[2]
    theta, phi = 2 * math.pi * u, math.acos(2 * v - 1) - math.pi / 2
    R = udp.R_START
    r0 = (R * math.cos(phi) * math.cos(theta), R * math.cos(phi) * math.sin(theta), R * math.sin(phi))
    t = T_START_MJD2000 + T0
    rP, vP = PLANETS[seq[0]].eph(t)
    v1, v_in = lambert(r0, rP, T0 * DAY2SEC, MU_TRAPPIST)
    arcs = [("entry", _arc(r0, v1, T0 * DAY2SEC, 2000))]
    events = [(seq[0], t, rP)]
    for k in range(6):
        beta, rp, eta, T = x[3 + 4 * k: 7 + 4 * k]
        i, j = seq[k], seq[k + 1]
        r_i, v_pl = PLANETS[i].eph(t)
        v_out = fb_prop(v_in, v_pl, rp * PLANETS[i].radius, beta, PLANETS[i].mu_self)
        arcs.append(("coast", _arc(r_i, v_out, eta * T * DAY2SEC)))
        r_d, _ = propagate_lagrangian(r_i, v_out, eta * T * DAY2SEC, MU_TRAPPIST)
        r_j, _ = PLANETS[j].eph(t + T)
        w1, v_in = lambert(r_d, r_j, (1 - eta) * T * DAY2SEC, MU_TRAPPIST)
        arcs.append(("lambert", _arc(r_d, w1, (1 - eta) * T * DAY2SEC)))
        t += T
        events.append((j, t, r_j))
    return arcs, events


def plot_trajectory(x, path, title=None, zoom=0.12):
    arcs, events = trajectory_arcs(x)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, lim, show_entry in ((axes[0], zoom, False), (axes[1], 10.5, True)):
        ax.scatter([0], [0], color="orange", s=60 if show_entry else 120, zorder=5, label="TRAPPIST-1")
        th = np.linspace(0, 2 * np.pi, 200)
        for k, p in enumerate(PLANETS):
            a = p.elements[0] / AU
            ax.plot(a * np.cos(th), a * np.sin(th), color=COLORS[k], lw=0.6, alpha=0.5)
        for kind, pts in arcs:
            if kind == "entry" and not show_entry:
                pts = pts[np.linalg.norm(pts, axis=1) < 1.5 * zoom]
                if len(pts) == 0:
                    continue
            style = {"entry": ("green", 1.2), "coast": ("royalblue", 0.8), "lambert": ("crimson", 0.8)}[kind]
            ax.plot(pts[:, 0], pts[:, 1], color=style[0], lw=style[1])
        for j, t, r in events:
            ax.scatter([r[0] / AU], [r[1] / AU], color=COLORS[j], s=40, zorder=6, edgecolor="k")
            if not show_entry:
                ax.annotate(NAMES[j], (r[0] / AU, r[1] / AU), xytext=(4, 4), textcoords="offset points", fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
    axes[0].set_title("Turi në sistemin planetar (zoom)")
    axes[1].set_title("Hyrja nga sfera 10 AU")
    axes[0].plot([], [], color="green", label="hyrja (Lambert)")
    axes[0].plot([], [], color="royalblue", label="lundrim pas fly-by")
    axes[0].plot([], [], color="crimson", label="hark Lambert pas DSM")
    axes[0].legend(loc="lower left", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fronts(series, path, title="Fronti Pareto (ΔV kundrejt kohës)"):
    """series: list of (label, points, style dict)."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for label, pts, st in series:
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            continue
        idx = nondominated(pts)
        if len(idx) == 0:
            continue
        P = pts[idx]
        xs = np.concatenate([P[:, 0], [REF[0]]])
        ys = np.concatenate([P[:, 1], [P[-1, 1]]])
        ax.step(xs, ys, where="post", **{k: v for k, v in st.items() if k != "marker"})
        ax.scatter(P[:, 0], P[:, 1], s=12, color=st.get("color"), marker=st.get("marker", "o"), label=label)
    ax.axvline(REF[0], color="k", ls=":", lw=0.8)
    ax.axhline(REF[1], color="k", ls=":", lw=0.8)
    ax.set_xlim(-50, REF[0] + 100)
    ax.set_xlabel("ΔV total [m/s]")
    ax.set_ylabel("Koha totale e misionit [ditë]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
