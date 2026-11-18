"""
Step 3 of the MPMCT -> NCT pipeline: after deletion + MR, reorder the MPMCT gates (using the
same commutation rule MR uses) so that gates that are cheap to convert *together* end up
adjacent, then convert in blocks exactly as in nct_experiments.py (E5).

Pipeline measured here:
    MPMCT (after deletion + MR)  ->  [step 3: pair-aware reordering]  ->  block-wise NCT conversion

Two partner scores are compared:
  rule    - geometric rule only, usable for any n:
            same target  -> Hamming distance between the two control patterns (1 is best)
            other target -> n + endpoint distance (paired only if nothing better is reachable)
  oracle  - exact pair cost from BFS (n = 3, 4 only), as an upper reference for the rule

Run:  python nct_study/nct_reorder.py [--quick]
Writes nct_study/results/reorder_report.md
"""

import os
import sys
import time
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nct_core import NCTOracle, gate_perm, circuit_perm
from nct_experiments import mpmct_circuit, block_dp, edge_vertices, canon_pair

QUICK = "--quick" in sys.argv
OUT = []


def emit(line=""):
    print(line, flush=True)
    OUT.append(line)


def table(headers, rows):
    emit("| " + " | ".join(headers) + " |")
    emit("|" + "---|" * len(headers))
    for r in rows:
        emit("| " + " | ".join(str(x) for x in r) + " |")
    emit()


# ---------------------------------------------------------------------------
# commutation + reordering
# ---------------------------------------------------------------------------

def commutes(a, b):
    """Full-control MPMCT gates are hypercube edges (transpositions); two of them commute
    iff they share no vertex (or are the same edge) - the same rule MR relies on."""
    va, vb = set(edge_vertices(a)), set(edge_vertices(b))
    return va == vb or not (va & vb)


def rule_score(a, b):
    n = len(a)
    va, vb = edge_vertices(a), edge_vertices(b)
    d = min(bin(x ^ y).count("1") for x in va for y in vb)
    if a.index(3) == b.index(3):
        return d                 # 1..n-1
    return n + d                 # always worse than any same-target partner


def make_oracle_score(oracle, n):
    cache = {}

    def score(a, b):
        # cost is invariant under line relabeling and pair reversal -> only 33 classes for n=4
        key = canon_pair(a, b) if n == 4 else (tuple(a), tuple(b))
        if key not in cache:
            a, b = [list(x) for x in key]
            p = gate_perm(b)[gate_perm(a)]
            c = oracle.cost(p)
            if n <= 3:  # single gates are convertible -> rank by saving
                c = c - oracle.cost(gate_perm(a)) - oracle.cost(gate_perm(b))
            cache[key] = 10 ** 6 if c is None else c
        return cache[key]
    return score


def reorder_for_pairs(gates, score):
    """
    Greedy left-to-right pairing. For gate i, look at every later gate j that can slide left
    to position i+1 (it commutes with every gate in between), take the one with the best
    score, move it next to i, and continue after the pair. Only commuting swaps are used, so
    the circuit's function is unchanged.
    """
    g = [list(x) for x in gates]
    i = 0
    while i < len(g) - 1:
        best = None
        for j in range(i + 1, len(g)):
            if all(commutes(g[k], g[j]) for k in range(i + 1, j)):
                s = score(g[i], g[j])
                if best is None or s < best[0]:
                    best = (s, j)
        if best is not None:
            g.insert(i + 1, g.pop(best[1]))
        i += 2
    return g


def adjacent_same_target(gates):
    return sum(1 for a, b in zip(gates, gates[1:]) if a.index(3) == b.index(3))


# ---------------------------------------------------------------------------

def run(n, probs, oracle, block_len, step, iters):
    rows = []
    stats = []
    for idx in probs:
        tab, circ, ok = mpmct_circuit(n, idx, iters, seed=idx)
        base_perm = list(circuit_perm(circ, n))
        variants = {"原順序 (MR 後)": circ,
                    "規則版重排": reorder_for_pairs(circ, rule_score),
                    "精確版重排": reorder_for_pairs(circ, make_oracle_score(oracle, n))}
        row = [f"{n}_{idx}", len(circ)]
        st = [f"{n}_{idx}"]
        for name, g in variants.items():
            assert list(circuit_perm(g, n)) == base_perm, name  # reordering must not change the function
            cost_fn = oracle.lookup if n <= 3 else oracle.cost
            pair, _ = block_dp(g, cost_fn, 2, step)
            blk, _ = block_dp(g, cost_fn, block_len, step)
            row += [pair if pair is not None else "無解", blk if blk is not None else "無解"]
            st.append(adjacent_same_target(g))
        rows.append(row)
        stats.append(st)
        print(f"  {n}_{idx} done", flush=True)
    hdr = ["題目", "MPMCT 閘數"]
    for name in ("原順序", "規則版", "精確版"):
        hdr += [f"{name} 兩兩", f"{name} 區塊≤{block_len}"]
    table(hdr, rows)
    emit("相鄰且目標相同的閘對數量 (重排效果的結構指標):")
    emit()
    table(["題目", "原順序", "規則版重排", "精確版重排"], stats)


def main():
    t0 = time.time()
    iters = 60 if QUICK else 300
    emit("# 第 3 步：為 NCT 重新排列 MPMCT 閘")
    emit()
    emit("流程：MPMCT (刪除 + MR 之後) → 重新排列 → 分組轉成 NCT。數字為 NCT 閘數 (越小越好)。")
    emit("電路與 E5 相同 (AE-QTS、" + str(iters) + " 次迭代、相同 seed)，所以「原順序」欄可以和 report.md 對照。")
    emit("所有重排後的電路都模擬驗證過功能不變。")
    emit()

    emit("## n=3")
    emit()
    o3 = NCTOracle(3, 20, verbose=False)
    run(3, (1, 2), o3, 3, 1, iters)

    emit("## n=4 (無輔助線；每組必須是偶數個閘；奇置換的 4_3、4_6 無解，略過)")
    emit()
    print("building n=4 BFS ball (depth 6)...", flush=True)
    o4 = NCTOracle(4, 6, verbose=False)
    o4.left_depth = 5
    probs = (1, 2, 8) if QUICK else (1, 2, 4, 5, 7, 8)
    run(4, probs, o4, 4 if QUICK else 6, 2, iters)

    emit(f"_總執行時間 {time.time() - t0:.0f}s_")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "reorder_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
