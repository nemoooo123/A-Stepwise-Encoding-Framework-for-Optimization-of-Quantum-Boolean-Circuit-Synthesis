"""
Does the framework's current fitness (MPMCT gate count) agree with the NCT cost?

For each problem, AE-QTS is run as usual, but EVERY neighbour circuit produced during the run is
recorded (deduplicated). Each circuit is converted with the general pipeline (nct_convert.py,
merge -> Barenco -> deletion/MR, plus 3-line windows for n = 3) and three numbers are kept:

  mpmct   : the current fitness (gate count after deletion + MR)
  proxy   : cheap NCT-aware fitness candidate = quantum cost of the merged circuit decomposed
            gate by gate, without NCT-level optimization (no peephole / windows)
  qc      : final quantum cost of the NCT circuit (NOT = CNOT = 1, TOF = 5)

Questions answered:
  1. rank correlation mpmct vs qc       (if high, changing the fitness is pointless)
  2. among the circuits with the minimum mpmct, how much does qc vary?
  3. how much lower is the best qc overall than the qc of the best-mpmct circuit? (potential gain)
  4. rank correlation proxy vs qc       (can the cheap proxy stand in for the real NCT cost?)

Run:  python nct_study/nct_fitness_study.py
Writes nct_study/results/fitness_study.md
"""

import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from nct_convert import convert_to_nct, merge_mpmct, decompose, quantum_cost, WindowTable
from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode, gen_nbrs
from utils.topology import decode_and_synthesize
from core.AE_QTS import updateQ

# (bits, idx, iterations, max circuits kept)
PROBLEMS = [(3, 1, 200, 3000), (3, 2, 200, 3000),
            (4, 1, 300, 4000), (4, 2, 300, 4000), (4, 8, 300, 4000), (4, 4, 300, 4000),
            (5, 7, 200, 1500)]


def collect_circuits(bits, idx, iters, seed):
    random.seed(seed)
    np.random.seed(seed)
    tab = DataLoader().get_output(bits, idx)
    cycles, _ = find_cycles(tab, check_zero_gate=True)
    q1, q2, q3, q4, enc, traj = build_encode(cycles)
    q1, q2, q3, q4 = (copy.deepcopy(x) for x in (q1, q2, q3, q4))
    seen = {}
    for _ in range(iters):
        n1, n2, n3, n4 = gen_nbrs(q1, q2, q3, q4, 50)
        sols = decode_and_synthesize(n1, n2, n3, n4, enc, bits, 50, traj)
        order = sorted(range(len(sols)), key=lambda i: sols[i][1])
        updateQ(q1, q2, q3, q4, 50, n1, n2, n3, n4, order, len(cycles), 0.005)
        for circ, _ in sols:
            key = tuple(tuple(g) for g in circ)
            if key not in seen:
                seen[key] = [list(g) for g in circ]
    return list(seen.values())


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    t0 = time.time()
    loader = DataLoader()
    t3 = WindowTable(3, 20)
    out = []

    def emit(s=""):
        print(s, flush=True)
        out.append(s)

    emit("# 適應度實驗：MPMCT 閘數 vs NCT 成本")
    emit()
    emit("每個題目跑一次 AE-QTS，收集過程中產生的**所有不重複電路**，每個都轉成 NCT 計算 quantum cost (QC)。")
    emit("proxy = 合併後逐閘拆解、不做 NCT 層最佳化的 QC (便宜，可當適應度候選)。")
    emit()
    hdr = ["題目", "電路數", "ρ(MPMCT, QC)", "MPMCT 最少", "其中 QC 範圍", "QC 最低的電路 (MPMCT / QC)",
           "可省 QC", "ρ(proxy, QC)"]
    emit("| " + " | ".join(hdr) + " |")
    emit("|" + "---|" * len(hdr))
    for bits, idx, iters, cap in PROBLEMS:
        ts = time.time()
        name = loader.names[bits][idx]
        circs = collect_circuits(bits, idx, iters, seed=idx)
        random.seed(0)
        if len(circs) > cap:
            # keep the best ones (they matter most) plus a random sample of the rest
            circs.sort(key=len)
            circs = circs[:cap // 2] + random.sample(circs[cap // 2:], cap - cap // 2)
        rows = []
        tables = (t3,) if bits == 3 else ()
        for c in circs:
            merged = merge_mpmct(c)
            pad = bits >= 4 and any(sum(1 for v in g if v in (0, 1)) == bits - 1 for g in merged)
            raw = []
            for g in merged:
                raw += decompose(g + [2] if pad else g)
            r = convert_to_nct(c, bits, tables=tables)
            rows.append((len(c), quantum_cost(raw), r["qc"], r["nct_gates"]))
        df = pd.DataFrame(rows, columns=["mpmct", "proxy", "qc", "nct"])
        m = df.mpmct.min()
        best_m = df[df.mpmct == m]
        best_q = df.loc[df.qc.idxmin()]
        gain = best_m.qc.min() - df.qc.min()
        emit(f"| {name} | {len(df)} | {spearman(df.mpmct, df.qc):.2f} | {m} | {best_m.qc.min()}~{best_m.qc.max()} "
             f"| {int(best_q.mpmct)} / {int(best_q.qc)} | {gain} ({100 * gain / best_m.qc.min():.0f}%) "
             f"| {spearman(df.proxy, df.qc):.2f} |")
        df.to_csv(os.path.join(HERE, "results", f"fitness_{name}.csv"), index=False)
        print(f"    ({time.time() - ts:.0f}s)", flush=True)
    emit()
    emit("欄位說明：")
    emit("- ρ = Spearman 等級相關 (1 = 排序完全一致)")
    emit("- 「其中 QC 範圍」= MPMCT 閘數同為最少的那些電路，轉成 NCT 後 QC 的最小~最大值")
    emit("- 「可省 QC」= (MPMCT 最少的電路中 QC 最低者) − (全部電路中 QC 最低者)")
    emit()
    emit(f"_總執行時間 {time.time() - t0:.0f}s_")
    with open(os.path.join(HERE, "results", "fitness_study.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
