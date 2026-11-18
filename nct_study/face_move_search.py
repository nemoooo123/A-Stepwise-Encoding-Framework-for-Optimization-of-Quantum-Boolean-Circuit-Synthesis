"""
Experiment: does letting the synthesis use gates with FEWER controls (a gate with k controls
moves 2^(n-1-k) parallel hypercube edges at once, i.e. a whole face/sub-cube) bring the NCT
result close to the optimum?

The SAME search is run with three gate sets, so only the gate set differs:
  full   - full-control MPMCT gates only (every other line is a 0/1 control): one edge per gate,
           i.e. the gate set of the current stepwise-encoding framework
  mpmct  - MPMCT gates with any number of controls (each other line is 0, 1 or "not used")
  nct    - strict NCT gates directly (<= 2 positive controls)

Search: beam search on the residual permutation r (start r = f, goal r = identity). A gate g is
appended at the output end, r' = g o r. States are ranked by the total Hamming distance
sum_x HD(x, r(x)) (the hypercube-distance guidance used in the group's CMC 2026 paper), ties broken
by the accumulated cost estimate. Result circuits of `full` / `mpmct` are converted with
nct_convert.convert_to_nct; every final circuit is verified by simulation.

Run:  python nct_study/face_move_search.py [--beam 5000]
Writes nct_study/results/face_move_report.md
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_core import gate_perm, nct_library, full_control_gates, pack
from nct_convert import (convert_to_nct, check, simulate, quantum_cost, optimize_nct, WindowTable,
                         mct_ncv_cost)
from utils.data_loader import DataLoader

PROBLEMS = [(3, 1), (3, 2), (4, 1), (4, 2), (4, 8)]
OPT = {(3, 1): (5, 13), (3, 2): (4, 8), (4, 1): (11, 23), (4, 2): (12, None), (4, 8): (12, None)}


def mpmct_all(n):
    gates = []
    for t in range(n):
        for vals in itertools.product((0, 1, 2), repeat=n - 1):
            g = list(vals)
            g.insert(t, 3)
            gates.append(g)
    return gates


def gate_cost(g):
    n = len(g)
    k = sum(1 for v in g if v in (0, 1))
    neg = sum(1 for v in g if v == 0)
    return mct_ncv_cost(k, n) + 2 * neg


def beam_search(tab, n, gates, beam, max_depth):
    N = 1 << n
    gp = np.stack([gate_perm(g) for g in gates]).astype(np.uint8)        # (G, N)
    gcost = np.array([gate_cost(g) for g in gates], dtype=np.int64)
    popc = np.array([bin(i).count("1") for i in range(N)], dtype=np.int64)
    ident = np.arange(N, dtype=np.uint8)

    res = np.asarray(tab, dtype=np.uint8)[None, :]
    cost = np.zeros(1, dtype=np.int64)
    parents, gidx = [], []
    for depth in range(1, max_depth + 1):
        G, B = gp.shape[0], res.shape[0]
        cand = gp[:, res].reshape(G * B, N)                                 # r' = g o r
        ccost = (cost[None, :] + gcost[:, None]).reshape(-1)
        h = popc[(cand ^ ident[None, :])].sum(1)
        par = np.tile(np.arange(B), G)
        gi = np.repeat(np.arange(G), B)
        # dedupe identical residuals, keep the cheapest
        order = np.lexsort((ccost, h))
        keys = pack(cand[order])
        _, first = np.unique(keys, return_index=True)
        keep = order[np.sort(first)]
        keep = keep[np.lexsort((ccost[keep], h[keep]))][:beam]
        parents.append(par[keep])
        gidx.append(gi[keep])
        res, cost = cand[keep], ccost[keep]
        done = np.where(h[keep] == 0)[0]
        if done.size:
            j = int(done[np.argmin(cost[done])])
            seq = []
            for lvl in range(depth - 1, -1, -1):
                seq.append(gates[gidx[lvl][j]])
                j = parents[lvl][j]
            # chosen gates were stacked at the output end: g_d o ... o g_1 o f = id,
            # so f = g_1 o ... o g_d, i.e. the circuit applies g_d first
            return [list(g) for g in seq]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=5000)
    ap.add_argument("--max-depth", type=int, default=80)
    args = ap.parse_args()

    loader = DataLoader()
    t3 = WindowTable(3, 20)
    out = []

    def emit(s=""):
        print(s, flush=True)
        out.append(s)

    emit("# 允許少控制的閘：能不能接近 NCT 最佳解？")
    emit()
    emit(f"同一個 beam search (寬度 {args.beam}，以 Hamming 距離總和引導)，只改變可用的閘。結果皆轉成嚴格 NCT 並模擬驗證。")
    emit("每格：`NCT 閘數 / QC / 輔助線`；括號內是搜尋直接得到的閘數。")
    emit()
    hdr = ["題目", "只用滿控制 (現在的框架)", "MPMCT 任意控制數", "直接用 NCT", "最佳解 (閘數 / QC)"]
    emit("| " + " | ".join(hdr) + " |")
    emit("|" + "---|" * len(hdr))
    for n, idx in PROBLEMS:
        name = loader.names[n][idx]
        tab = loader.get_output(n, idx)
        cells = []
        for kind, gates in (("full", full_control_gates(n)), ("mpmct", mpmct_all(n)), ("nct", nct_library(n))):
            t0 = time.time()
            seq = beam_search(tab, n, gates, args.beam, args.max_depth)
            if seq is None:
                cells.append("未找到")
                continue
            assert list(simulate(seq, n)) == list(tab), (name, kind)
            if kind == "nct":
                nct = optimize_nct(seq, [t3])
                assert list(simulate(nct, n)) == list(tab)
                cells.append(f"{len(nct)} / {quantum_cost(nct)} / 0 ({len(seq)})")
            else:
                r = convert_to_nct(seq, n, tables=[t3])
                assert check(seq, n, r)
                cells.append(f"{r['nct_gates']} / {r['qc']} / {r['ancilla']} ({len(seq)})")
            print(f"  {name} {kind}: {cells[-1]}  [{time.time() - t0:.1f}s]", flush=True)
        o = OPT.get((n, idx))
        cells.append(f"{o[0]} / {o[1] if o[1] else '?'}" if o else "-")
        emit(f"| {name} | " + " | ".join(cells) + " |")
    emit()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "face_move_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
