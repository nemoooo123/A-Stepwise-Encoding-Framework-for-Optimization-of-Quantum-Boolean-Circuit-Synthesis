"""
"Advanced moving rule": unlock merges that ordinary MR cannot reach.

Background identities (all gates are involutions, A.A = I):
  (1) conjugation:   A then B  ==  B then (B A B)
      -> if A, B commute this is the ordinary moving rule; if they share a state, B A B is a swap
         of two states at Hamming distance 2, which is NOT a single gate (moving costs +2 gates).
  (2) merge:         C(S, x=0; t) then C(S, x=1; t) == C(S; t)          (one control fewer)
  (3) 3 -> 2 rewrite (derived from (1)/(2)): for a window  A, B, C  with A and C mergeable into
      the face gate F and B blocking them, the window's permutation W is fixed, so
          W = X o F  ->  X = W o F        (window becomes  F, X)
          W = F o X  ->  X = F o W        (window becomes  X, F)
      (F is an involution). If X is a single MPMCT gate (possibly with fewer controls), the
      three gates are replaced by two.

Pass: repeat { ordinary merge (nct_convert.merge_mpmct); 3 -> 2 rewrites } until nothing changes.
Every rewrite is checked by simulation of the window; the final circuit is checked against the
original on all inputs.

Run:  python nct_study/advanced_mr.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_convert import merge_mpmct, commute, _merge, convert_to_nct, simulate, WindowTable
from nct_experiments import mpmct_circuit
from utils.data_loader import DataLoader


def perm_of(gates, n):
    return simulate(gates, n)


def as_single_gate(p, n):
    """If permutation p (array over 2^n states) is one MPMCT gate (any controls), return it."""
    moved = np.nonzero(p != np.arange(p.size))[0]
    if moved.size == 0:
        return "identity"
    diffs = np.unique(moved ^ p[moved])
    if diffs.size != 1 or bin(int(diffs[0])).count("1") != 1:
        return None
    tbit = int(diffs[0]).bit_length() - 1
    low = moved[(moved >> tbit) & 1 == 0]
    # low must be a full sub-cube over the non-target bits: constant bits = controls
    g = [2] * n
    g[n - 1 - tbit] = 3
    varying = 0
    for j in range(n):
        if j == tbit:
            continue
        vals = np.unique((low >> j) & 1)
        if vals.size == 1:
            g[n - 1 - j] = int(vals[0])
        else:
            varying += 1
    if low.size != 1 << varying:
        return None
    # check the gate really equals p
    if not np.array_equal(perm_of([g], n), p):
        return None
    return g


def try_rewrites(g, n):
    """One sweep of 3 -> 2 rewrites. Returns (new circuit, number of rewrites)."""
    count = 0
    i = 0
    while i < len(g):
        done = False
        for j in range(i + 2, min(len(g), i + 40)):
            if _merge(g[i], g[j]) is None or _merge(g[i], g[j])[0] != "merge":
                continue
            between = list(range(i + 1, j))
            blockers = [k for k in between if not commute(g[k], g[j])]
            if len(blockers) != 1:
                continue
            b = blockers[0]
            # A must be able to slide right up to B (commute with everything in (i, b))
            if not all(commute(g[i], g[k]) for k in range(i + 1, b)):
                continue
            A, B, C = g[i], g[b], g[j]
            F = _merge(A, C)[1]
            W = perm_of([A, B, C], n)
            pf = perm_of([F], n)
            for order in ("FX", "XF"):
                X = W[pf] if order == "FX" else pf[W]       # X = W o F   or   X = F o W
                xg = as_single_gate(X, n)
                if xg is None:
                    continue
                new = [F] if xg == "identity" else ([F, xg] if order == "FX" else [xg, F])
                assert np.array_equal(perm_of(new, n), W)
                # rebuild: gates in (i, b) that commute with A stay before the window,
                # gates in (b, j) commute with C and stay after it
                left = [g[k] for k in range(i + 1, b)]
                right = [g[k] for k in range(b + 1, j)]
                g = g[:i] + left + new + right + g[j + 1:]
                count += 1
                done = True
                break
            if done:
                break
        if not done:
            i += 1
    return g, count


def advanced_merge(circ, n):
    g = merge_mpmct(circ)
    total = 0
    while True:
        g, c = try_rewrites(g, n)
        total += c
        g2 = merge_mpmct(g)
        if c == 0 and len(g2) == len(g):
            return g2, total
        g = g2


def main():
    t0 = time.time()
    loader = DataLoader()
    t3 = WindowTable(3, 20)
    print(f"{'problem':16s} {'MPMCT':>6s} {'MR+merge':>9s} {'advanced':>9s} {'rewrites':>8s} | "
          f"{'NCT (merge)':>14s} {'NCT (advanced)':>15s}")
    for n, idx in [(4, 1), (4, 2), (4, 8), (4, 4), (5, 7), (5, 6), (6, 8)]:
        name = loader.names[n][idx]
        tab, circ, _ = mpmct_circuit(n, idx, 300 if n <= 5 else 200, seed=idx)
        base = merge_mpmct(circ)
        adv, rw = advanced_merge(circ, n)
        assert np.array_equal(perm_of(adv, n), perm_of(circ, n))
        r0 = convert_to_nct(base, n, tables=[t3], merge=False)
        r1 = convert_to_nct(adv, n, tables=[t3], merge=False)
        ctrl = lambda gs: np.mean([sum(v in (0, 1) for v in x) for x in gs])
        print(f"{name:16s} {len(circ):6d} {len(base):9d} {len(adv):9d} {rw:8d} | "
              f"{r0['nct_gates']:5d} / QC {r0['qc']:5d}  {r1['nct_gates']:5d} / QC {r1['qc']:5d}   "
              f"avg controls {ctrl(base):.2f} -> {ctrl(adv):.2f}", flush=True)
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
