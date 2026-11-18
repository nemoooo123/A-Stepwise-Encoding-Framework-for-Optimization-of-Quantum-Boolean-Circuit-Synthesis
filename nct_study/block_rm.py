"""
Same-target block resynthesis with fixed-polarity Reed-Muller (FPRM) expansions.

Gates with the same target t always commute, and a block of them computes
        t ^= g(other lines),   g = XOR of the blocks' control cubes.
Re-expressing g as an XOR of positive-literal products (after complementing a chosen set of
control lines, i.e. a fixed polarity) gives an equivalent block:
        NOT(lines in polarity) ; one gate per RM term (positive controls) ; NOT(lines in polarity)
This single rule covers
        x.f ^ ~x.f = f                (2 gates -> 1 with one control fewer)
        4 cubes -> a CNOT / NOT       (whole sub-cube)
        ab ^ a~b ^ ~ab = 1 ^ ~a~b     (3 gates -> NOT + the "missing" gate, via complemented polarity)
All 2^(n-1) polarities are tried; the cheapest (estimated NCT cost after decomposition) is kept,
or the original block if nothing is cheaper.

Blocks are formed with the ordinary moving rule (a later same-target gate joins the block if it
commutes with every gate in between).

Run:  python nct_study/block_rm.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_convert import (commute, merge_mpmct, decompose, convert_to_nct, simulate, quantum_cost,
                         WindowTable, NeedsAncilla)
from nct_experiments import mpmct_circuit
from utils.data_loader import DataLoader


def gather_blocks(gates):
    """Greedy MR grouping: returns a list of blocks (lists of gates), order-preserving."""
    g = [list(x) for x in gates]
    blocks = []
    used = [False] * len(g)
    for i in range(len(g)):
        if used[i]:
            continue
        t = g[i].index(3)
        block = [g[i]]
        used[i] = True
        between = []                       # gates skipped over (must all commute with joiners)
        for j in range(i + 1, len(g)):
            if used[j]:
                continue
            if g[j].index(3) == t and all(commute(h, g[j]) for h in between):
                block.append(g[j])
                used[j] = True
            else:
                between.append(g[j])
                # stop when nothing later could still join (cheap cut-off)
                if len(between) > 60:
                    break
        blocks.append(block)
    return blocks


def control_function(block, t, n):
    """Truth vector of g over the n-1 non-target lines (index bits in line order, MSB first)."""
    others = [l for l in range(n) if l != t]
    m = len(others)
    vec = np.zeros(1 << m, dtype=np.uint8)
    xs = np.arange(1 << m)
    for gate in block:
        sel = np.ones(1 << m, dtype=bool)
        for k, l in enumerate(others):
            v = gate[l]
            if v in (0, 1):
                sel &= ((xs >> (m - 1 - k)) & 1) == v
        vec ^= sel.astype(np.uint8)
    return vec, others


def rm_transform(vec):
    """Positive-polarity Reed-Muller coefficients (binary Moebius transform)."""
    a = vec.copy()
    m = int(np.log2(a.size))
    for k in range(m):
        step = 1 << k
        for s in range(0, a.size, 2 * step):
            a[s + step:s + 2 * step] ^= a[s:s + step]
    return a


def gate_cost(gate):
    try:
        d = decompose(gate)
    except NeedsAncilla:
        d = decompose(gate + [2])          # would need the ancilla; cost it with one
    return len(d), quantum_cost(d)


def resynth_block(block, n):
    t = block[0].index(3)
    vec, others = control_function(block, t, n)
    m = len(others)
    orig = sum(gate_cost(x)[0] for x in block)
    best = (orig, block)
    xs = np.arange(1 << m)
    for pol in range(1 << m):
        # g'(y) = g(y ^ pol): complement the lines whose polarity bit is 1
        coeffs = rm_transform(vec[xs ^ pol])
        gates = []
        neg_lines = [others[k] for k in range(m) if pol >> (m - 1 - k) & 1]
        nots = []
        for l in neg_lines:
            x = [2] * n
            x[l] = 3
            nots.append(x)
        for term in np.nonzero(coeffs)[0]:
            x = [2] * n
            x[t] = 3
            for k in range(m):
                if term >> (m - 1 - k) & 1:
                    x[others[k]] = 1
            gates.append(x)
        cand = nots + gates + nots
        cost = sum(gate_cost(x)[0] for x in cand)
        if cost < best[0]:
            best = (cost, cand)
    return best[1]


def block_rm(gates, n):
    out = []
    for b in gather_blocks(merge_mpmct(gates)):
        out += resynth_block(b, n)
    return out


def main():
    t0 = time.time()
    loader = DataLoader()
    t3 = WindowTable(3, 20)
    print(f"{'problem':16s} {'MPMCT':>5s} | {'current (merge->NCT)':>22s} | {'+ same-target RM':>20s} | "
          f"{'blocks>1':>8s} {'avg size':>8s}")
    for n, idx in [(4, 1), (4, 2), (4, 8), (4, 4), (4, 3), (5, 7), (5, 6), (6, 8)]:
        name = loader.names[n][idx]
        tab, circ, _ = mpmct_circuit(n, idx, 300 if n <= 5 else 200, seed=idx)
        r0 = convert_to_nct(circ, n, tables=[t3])
        new = block_rm(circ, n)
        assert np.array_equal(simulate(new, n), simulate(circ, n))
        r1 = convert_to_nct(new, n, tables=[t3])
        blocks = gather_blocks(merge_mpmct(circ))
        multi = [b for b in blocks if len(b) > 1]
        print(f"{name:16s} {len(circ):5d} | {r0['nct_gates']:6d} gates / QC {r0['qc']:5d} | "
              f"{r1['nct_gates']:5d} gates / QC {r1['qc']:5d} | {len(multi):8d} "
              f"{np.mean([len(b) for b in blocks]):8.2f}", flush=True)
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
