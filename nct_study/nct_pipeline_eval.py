"""
Evaluate the general MPMCT -> NCT pipeline (nct_convert.py) on the circuits AE-QTS produces.

Columns (NCT gate count / quantum cost with NOT=CNOT=1, TOF=5):
  A  baseline     : gate-wise Barenco decomposition + deletion/MR
  B  +merge       : MPMCT-level merging first
  C  +window3     : B + exact 3-line window resynthesis
  D  +window4     : C + 4-line window resynthesis (optimum <= 6 table)
References: exact strict-NCT optimum (n <= 4, no ancilla) and Saeedi et al. 2010 (JETC, Table 2,
best of their hybrid framework / previous best, quantum cost, no ancilla).

Run:  python nct_study/nct_pipeline_eval.py [--quick]
Writes nct_study/results/pipeline_report.md
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_core import parity
from nct_convert import convert_to_nct, check, WindowTable
from nct_experiments import mpmct_circuit
from utils.data_loader import DataLoader

QUICK = "--quick" in sys.argv

OPT = {(3, 1): 5, (3, 2): 4, (4, 1): 11, (4, 2): 12, (4, 8): 12,
       (4, 4): "≥15", (4, 5): "≥15", (4, 7): "≥15"}
SAEEDI_QC = {"nth_prime3_inc": 6, "hwb4": 23, "4_49": 32, "nth_prime4_inc": 51, "hwb5": 91,
             "nth_prime5_inc": 91, "hwb6": 107, "nth_prime6_inc": 667, "permanent2x2": 47}


def main():
    t0 = time.time()
    out = []

    def emit(s=""):
        print(s, flush=True)
        out.append(s)

    loader = DataLoader()
    print("building window tables...", flush=True)
    t3 = WindowTable(3, 20)
    t4 = WindowTable(4, 5 if QUICK else 6)
    stages = [("A 基準", dict(merge=False, tables=())),
              ("B +合併", dict(merge=True, tables=())),
              ("C +3線視窗", dict(merge=True, tables=(t3,))),
              ("D +4線視窗", dict(merge=True, tables=(t3, t4)))]

    emit("# MPMCT → 嚴格 NCT 通用轉換流程評估")
    emit()
    emit("電路來自 AE-QTS (seed 固定)。每格為 `NCT 閘數 / quantum cost`。所有結果皆經模擬驗證功能正確。")
    emit("「輔助線」= 合併後仍有用滿所有線的閘 (n−1 個控制)，因此整個電路加 1 條 dirty ancilla。")
    emit()
    hdr = ["題目", "n", "奇偶", "MPMCT", "合併後", "輔助線"] + [s for s, _ in stages] + \
          ["NCT 最佳閘數", "Saeedi 2010 QC"]
    emit("| " + " | ".join(hdr) + " |")
    emit("|" + "---|" * len(hdr))
    bits_list = (3, 4, 5) if QUICK else (3, 4, 5, 6)
    for bits in bits_list:
        iters = 300 if bits <= 5 else 200
        for idx in sorted(loader.data_map[bits]):
            name = loader.names[bits][idx]
            ts = time.time()
            tab, circ, ok = mpmct_circuit(bits, idx, iters, seed=idx)
            cells, merged, anc = [], None, None
            for label, kw in stages:
                r = convert_to_nct(circ, bits, **kw)
                assert check(circ, bits, r), (name, label)
                if label.startswith("B"):
                    merged, anc = r["mpmct_after_merge"], r["ancilla"]
                cells.append(f"{r['nct_gates']} / {r['qc']}")
            row = [name, bits, "奇" if parity(tab) else "偶", len(circ), merged, anc] + cells + \
                  [OPT.get((bits, idx), "-"), SAEEDI_QC.get(name, "-")]
            emit("| " + " | ".join(str(x) for x in row) + " |")
            print(f"    ({time.time() - ts:.0f}s)", flush=True)
    emit()
    emit(f"_總執行時間 {time.time() - t0:.0f}s_")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "pipeline_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
