"""
Experiment: drop Tsai's "return" gates and iterate.

For a residual r (output-side synthesis, goal r = identity), an element x with r(x) = y != x is
fixed by walking the OUTPUT value y to x one bit at a time with full-control MPMCT gates (each gate
= one hypercube edge, i.e. a swap of two adjacent states):

  tsai     - walk y -> ... -> x (l gates) and walk the intermediate nodes back (l-1 gates), so that
             ONLY the pair moves. One pass over the cycles always finishes.
  forward  - walk y -> ... -> x only (l gates). The rows whose outputs sat on the intermediate nodes
             are displaced; they are left for the next round. Rounds repeat (cycles recomputed from
             the residual) until total Hamming distance H = 0, or a round limit is hit.
  forward+avoid - same, but the order of bit flips is chosen so that the walk avoids intermediate
             nodes that are currently correct outputs (already-fixed rows) whenever possible.

Gate counts are measured after deletion + moving rule; the MPMCT circuits are also converted to
strict NCT (nct_convert) to report NCT gates / QC. Every circuit is verified by simulation.

Run:  python nct_study/forward_only_iteration.py
"""

import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_convert import convert_to_nct, peephole, simulate, WindowTable
from nct_core import parity
from utils.data_loader import DataLoader


def cycles_of(r):
    seen = [False] * len(r)
    out = []
    for s in range(len(r)):
        if seen[s] or r[s] == s:
            seen[s] = True
            continue
        c, x = [], s
        while not seen[x]:
            seen[x] = True
            c.append(x)
            x = int(r[x])
        out.append(c)
    return out


class Synth:
    def __init__(self, tab, n):
        self.n = n
        self.r = np.array(tab, dtype=np.int64)
        self.inv = np.empty_like(self.r)
        self.inv[self.r] = np.arange(self.r.size)
        self.gates = []  # output-side order

    def swap_edge(self, v, bit):
        """Full-control MPMCT gate swapping output values v and v ^ (1 << bit)."""
        n = self.n
        w = v ^ (1 << bit)
        g = [(v >> (n - 1 - i)) & 1 for i in range(n)]
        g[n - 1 - bit] = 3
        self.gates.append(g)
        a, b = self.inv[v], self.inv[w]            # rows whose outputs are v and w
        self.r[a], self.r[b] = w, v
        self.inv[v], self.inv[w] = b, a

    def bit_order(self, y, x, avoid):
        bits = [j for j in range(self.n) if (x ^ y) >> j & 1]
        if not avoid:
            return bits
        # greedy: next flip prefers a node that is NOT a correct (fixed) output, i.e. r[node] != node
        order, cur = [], y
        rest = list(bits)
        while rest:
            def bad(j):
                nxt = cur ^ (1 << j)
                return nxt != x and self.r[nxt] == nxt  # would displace a fixed row
            rest.sort(key=bad)
            j = rest.pop(0)
            order.append(j)
            cur ^= 1 << j
        return order

    def fix(self, x, mode):
        y = int(self.r[x])
        if y == x:
            return
        order = self.bit_order(y, x, mode == "forward+avoid")
        path = [y]
        for j in order:
            path.append(path[-1] ^ (1 << j))
        # walk the value y to x: swap along the path
        for k, j in enumerate(order):
            self.swap_edge(path[k], j)
        if mode == "tsai":
            # walk the intermediates back so that only the pair (x, original row) moved
            for k in range(len(order) - 2, -1, -1):
                self.swap_edge(path[k], order[k])

    def run(self, mode, max_rounds=200):
        rounds = 0
        while not np.array_equal(self.r, np.arange(self.r.size)):
            rounds += 1
            if rounds > max_rounds:
                return None, rounds
            for cyc in cycles_of(self.r):
                for x in cyc:
                    self.fix(x, mode)
        return rounds, rounds


def evaluate(tab, n, mode, t3):
    s = Synth(tab, n)
    ok, rounds = s.run(mode)
    if ok is None:
        return None
    circ = list(reversed(s.gates))
    assert list(simulate(circ, n)) == list(tab)
    mp = peephole(circ)
    conv = convert_to_nct(mp, n, tables=[t3])
    return {"rounds": rounds, "mpmct": len(mp), "nct": conv["nct_gates"], "qc": conv["qc"],
            "anc": conv["ancilla"]}


def main():
    t0 = time.time()
    t3 = WindowTable(3, 20)
    modes = ("tsai", "forward", "forward+avoid")
    print("== random permutations: convergence and average size ==")
    rng = np.random.default_rng(0)
    for n, cnt in ((3, 2000), (4, 300), (5, 100), (6, 30), (7, 10)):
        perms = [rng.permutation(1 << n) for _ in range(cnt)]
        line = [f"n={n} ({cnt})"]
        for mode in modes:
            res = [evaluate(p, n, mode, t3) for p in perms]
            ok = [r for r in res if r]
            fail = cnt - len(ok)
            if ok:
                line.append(f"{mode}: fail {fail}, rounds {np.mean([r['rounds'] for r in ok]):.1f}, "
                            f"MPMCT {np.mean([r['mpmct'] for r in ok]):.1f}, NCT {np.mean([r['nct'] for r in ok]):.1f}, "
                            f"QC {np.mean([r['qc'] for r in ok]):.0f}")
            else:
                line.append(f"{mode}: fail {fail}")
        print(" | ".join(line), flush=True)

    print("\n== catalog (MPMCT gates / NCT gates / QC / rounds) ==")
    loader = DataLoader()
    for n in (3, 4, 5, 6):
        for idx in sorted(loader.data_map[n]):
            tab = loader.get_output(n, idx)
            cells = []
            for mode in modes:
                r = evaluate(tab, n, mode, t3)
                cells.append(f"{r['mpmct']} / {r['nct']} / {r['qc']} / {r['rounds']}r" if r else "no convergence")
            print(f"{loader.names[n][idx]:16s} " + " | ".join(f"{m}: {c}" for m, c in zip(modes, cells)),
                  flush=True)
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
