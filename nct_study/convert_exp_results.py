"""
Convert the best MPMCT circuits saved by main.py (exp/<n>_bit/<ALGO>_Results/<ALGO>_<n>_<idx>.txt)
into strict NCT and tabulate them against the references.

Usage (from the repo root):
    python nct_study/convert_exp_results.py                 # every result file under exp/
    python nct_study/convert_exp_results.py --algo AE-QTS   # only one algorithm
    python nct_study/convert_exp_results.py --window4       # also 4-line window resynthesis
    python nct_study/convert_exp_results.py --verify-max 12 # simulate-check circuits up to 12 lines
    python nct_study/convert_exp_results.py --exp-dir D:/runs/exp   # read results from elsewhere

Output: nct_study/results/exp_nct_table.md and exp_nct_table.csv

Columns
  MPMCT       gate count of the saved circuit (= the framework's fitness)
  merged      MPMCT gates after MPMCT-level merging
  anc         extra (dirty) ancilla lines used by the conversion (0 or 1)
  NCT         strict NCT gate count after conversion + NCT-level optimization
  QC(NCT)     quantum cost of that NCT circuit, NOT = CNOT = 1, TOF = 5
  QC(NCV)     quantum cost with the NCV-level formulas used by Saeedi et al. 2010 (computed on the
              merged MPMCT circuit) -> the column to compare with Saeedi
  NCT opt     exact strict-NCT optimum without ancilla (n <= 4; from nct_optimal_benchmarks.py)
  Saeedi QC   Saeedi et al., ACM JETC 2010, Table 2: min(previous best, their hybrid framework)
"""

import argparse
import ast
import csv
import glob
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from nct_convert import convert_to_nct, check, simulate, ncv_quantum_cost, merge_mpmct, WindowTable
from nct_core import parity
from utils.data_loader import DataLoader

NCT_OPT = {"f3_1": 5, "nth_prime3_inc": 4, "hwb4": 11, "4_49": 12, "nth_prime4_inc": 12,
           "4b15g_2": "≥15", "4b15g_3": "≥15", "4b15g_5": "≥15"}

SAEEDI_QC = {
    "nth_prime3_inc": 6, "4_49": 32, "hwb4": 23, "nth_prime4_inc": 51,
    "hwb5": 91, "nth_prime5_inc": 91, "hwb6": 107, "nth_prime6_inc": 667, "permanent2x2": 47,
    "hwb7": 2611, "hwb8": 6940, "hwb9": 16173, "hwb10": 35618, "hwb11": 90745, "hwb12": 198928,
    "hwb13": 436305, "hwb14": 994340, "hwb15": 1999194, "hwb16": 4730024,
    "nth_prime7_inc": 2695, "nth_prime8_inc": 7618, "nth_prime9_inc": 17975,
    "nth_prime10_inc": 40299, "nth_prime11_inc": 95431, "nth_prime12_inc": 208227,
    "nth_prime13_inc": 474660, "nth_prime14_inc": 1018661, "nth_prime15_inc": 2271370,
    "nth_prime16_inc": 4823320, "nth_prime17_inc": 10592640,
    "ham7": 49, "permanent3x3": 1884,
}

FILE_RE = re.compile(r"^(?P<algo>.+)_(?P<n>\d+)_(?P<idx>\d+)\.txt$")


def read_best_circuit(path):
    with open(path, "r", encoding="utf-16") as f:
        text = f.read()
    m = re.search(r"Best Circuit Structure:\s*(\[.*?\])\s*\n=+", text, re.S)
    if not m:
        raise ValueError("no 'Best Circuit Structure' block")
    return [list(g) for g in ast.literal_eval(m.group(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=None)
    ap.add_argument("--window4", action="store_true")
    ap.add_argument("--verify-max", type=int, default=12,
                    help="simulate-check conversions with at most this many lines")
    ap.add_argument("--exp-dir", default=os.path.join(ROOT, "exp"),
                    help="folder written by main.py (default: <repo>/exp)")
    args = ap.parse_args()

    loader = DataLoader()
    files = sorted(glob.glob(os.path.join(args.exp_dir, "*_bit", "*_Results", "*.txt")))
    if not files:
        print("No result files under exp/. Run main.py first.")
        return
    tables = [WindowTable(3, 20)]
    if args.window4:
        print("building 4-line window table (~1.5 min)...", flush=True)
        tables.append(WindowTable(4, 6))

    rows = []
    for path in files:
        fm = FILE_RE.match(os.path.basename(path))
        if not fm:
            continue
        algo, n, idx = fm["algo"], int(fm["n"]), int(fm["idx"])
        if args.algo and algo != args.algo:
            continue
        if idx == 0:
            print(f"skip {path}: custom sparse problem (idx 0) has no catalog table")
            continue
        name = loader.names.get(n, {}).get(idx, f"{n}_{idx}")
        tab = loader.get_output(n, idx)
        t0 = time.time()
        circ = read_best_circuit(path)

        # sanity: the saved circuit must realise the catalog function (or its inverse)
        if n <= 16:
            perm = simulate(circ, n)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(perm.size)
            if not (list(perm) == tab or list(inv) == tab):
                print(f"WARNING {path}: saved circuit does not realise {name}; skipped")
                continue

        merged = merge_mpmct(circ)
        r = convert_to_nct(circ, n, tables=tables)
        verified = "✓" if r["lines"] <= args.verify_max and check(circ, n, r) else \
                   ("-" if r["lines"] > args.verify_max else "✗")
        if verified == "✗":
            print(f"ERROR {path}: NCT circuit is not equivalent!")
        ncv = ncv_quantum_cost([g + [2] for g in merged] if r["ancilla"] else merged, r["lines"])
        rows.append({
            "algo": algo, "n": n, "name": name, "parity": "odd" if parity(tab) else "even",
            "mpmct": len(circ), "merged": r["mpmct_after_merge"], "anc": r["ancilla"],
            "nct": r["nct_gates"], "qc_nct": r["qc"], "qc_ncv": ncv,
            "nct_opt": NCT_OPT.get(name, "-"), "saeedi_qc": SAEEDI_QC.get(name, "-"),
            "verified": verified,
        })
        print(f"{algo} {name}: MPMCT {len(circ)} -> NCT {r['nct_gates']} (QC {r['qc']}, NCV-QC {ncv}) "
              f"[{time.time() - t0:.1f}s]", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    csv_path = os.path.join(HERE, "results", "exp_nct_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = ["演算法", "n", "題目", "奇偶", "MPMCT", "合併後", "輔助線", "NCT 閘數", "QC (NCT)",
           "QC (NCV 公式)", "NCT 最佳", "Saeedi QC", "驗證"]
    lines = ["# 正式實驗結果轉成嚴格 NCT", "",
             "QC (NCT)：NOT = CNOT = 1、Toffoli = 5。QC (NCV 公式)：與 Saeedi et al. 2010 相同的計算方式，用來和它比較。",
             "Saeedi QC 取其 Table 2 中「先前最佳」與「其混合框架」的較小值；該方法不使用輔助線。", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in
                     ["algo", "n", "name", "parity", "mpmct", "merged", "anc", "nct", "qc_nct", "qc_ncv",
                      "nct_opt", "saeedi_qc", "verified"]).replace("| odd |", "| 奇 |").replace("| even |", "| 偶 |")
                     + " |")
    with open(os.path.join(HERE, "results", "exp_nct_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {csv_path} and exp_nct_table.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
