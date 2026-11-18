"""
Exact strict-NCT optimum (no ancilla) for every 3-bit and 4-bit catalog problem, as the
ground truth to compare any conversion pipeline against.

  n = 3: exact over the whole group.
  n = 4: meet in the middle over a depth-6 BFS ball -> exact whenever the optimum is <= 12;
         otherwise reported as ">12" (a lower bound). Odd permutations are unreachable.

Run:  python nct_study/nct_optimal_benchmarks.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_core import NCTOracle, parity, nct_library, gate_str
from nct_experiments import synth
from utils.data_loader import DataLoader


def main():
    loader = DataLoader()
    o3 = NCTOracle(3, 20, verbose=False)
    print("| n | idx | 名稱 | 奇偶 | 嚴格 NCT 最佳閘數 |")
    print("|---|---|---|---|---|")
    for idx, tab in sorted(loader.data_map[3].items()):
        p = np.array(tab, dtype=np.uint8)
        print(f"| 3 | {idx} | {loader.names[3][idx]} | {'奇' if parity(p) else '偶'} | {o3.lookup(p)} |")
        print("  ", " ; ".join(gate_str(g) for g in synth(o3, p)))
    o4 = NCTOracle(4, 6, verbose=False)
    for idx, tab in sorted(loader.data_map[4].items()):
        p = np.array(tab, dtype=np.uint8)
        if parity(p):
            res = "無解 (奇置換，需輔助線)"
        else:
            c = o4.cost(p)
            res = str(c) if c is not None else ">12"
        print(f"| 4 | {idx} | {loader.names[4][idx]} | {'奇' if parity(p) else '偶'} | {res} |", flush=True)


if __name__ == "__main__":
    main()
