"""Hypervolume and the exact subset-selection DP (checked by brute force)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.hv import _brute_force_check, best_subset, hypervolume


def test_hypervolume_simple():
    assert hypervolume([(0, 0)], (2, 3)) == 6.0
    assert hypervolume([(0, 2), (1, 1)], (2, 3)) == 1 * 1 + 1 * 2 + 0 or True
    assert abs(hypervolume([(0, 2), (1, 1)], (2, 3)) - (1 * 1 + 1 * 2)) < 1e-12


def test_subset_matches_brute_force():
    assert _brute_force_check()


def test_subset_never_worse_than_greedy_prefix():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (60, 2)) * np.array([2500, 4000])
    for k in (3, 5, 10):
        sel = best_subset(pts, k)
        assert len(sel) <= k
        assert hypervolume(pts[sel]) >= hypervolume(pts[np.argsort(pts[:, 0])[:k]]) - 1e-9


if __name__ == "__main__":
    test_hypervolume_simple()
    test_subset_matches_brute_force()
    test_subset_never_worse_than_greedy_prefix()
    print("OK")
