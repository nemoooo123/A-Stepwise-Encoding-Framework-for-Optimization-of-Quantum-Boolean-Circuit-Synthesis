"""
Synthesize strict-NCT circuits with the Hamming-guided NCT beam search (face_move_search.py),
verify them on every input, and export them:
  results/circuits/<name>.txt   gate list in the framework's 0/1/2/3 format + readable form + truth table
  results/circuits/<name>.tfc   RevLib .tfc format (t1 = NOT, t2 = CNOT, t3 = Toffoli)

Run:  python nct_study/export_nct_circuits.py [--beam 20000] [--problems 4:2 4:1]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_core import nct_library, gate_str
from nct_convert import optimize_nct, quantum_cost, simulate, WindowTable
from face_move_search import beam_search
from utils.data_loader import DataLoader


def to_tfc(circ, n, name):
    names = [chr(ord("a") + i) for i in range(n)]
    lines = [f"# {name}: strict NCT, {len(circ)} gates, QC {quantum_cost(circ)}",
             f".v {','.join(names)}", f".i {','.join(names)}", f".o {','.join(names)}", "BEGIN"]
    for g in circ:
        ctrl = [names[i] for i, v in enumerate(g) if v == 1]
        lines.append(f"t{len(ctrl) + 1} {','.join(ctrl + [names[g.index(3)]])}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=20000)
    ap.add_argument("--problems", nargs="*", default=["4:2", "4:1", "4:8"])
    a = ap.parse_args()
    loader = DataLoader()
    t3 = WindowTable(3, 20)
    outdir = os.path.join(HERE, "results", "circuits")
    os.makedirs(outdir, exist_ok=True)
    for spec in a.problems:
        n, idx = map(int, spec.split(":"))
        name = loader.names[n][idx]
        tab = loader.get_output(n, idx)
        seq = beam_search(tab, n, nct_library(n), a.beam, 80)
        circ = optimize_nct(seq, [t3])
        got = list(simulate(circ, n))
        ok = got == list(tab)
        print(f"\n=== {name} ({n} lines): {len(circ)} gates, QC {quantum_cost(circ)}, verified: {ok} ===")
        for k, g in enumerate(circ, 1):
            print(f"  {k:2d}. {gate_str(g):24s} {g}")
        rows = [f"  {x:0{n}b} -> {y:0{n}b}   (target {t:0{n}b}) {'OK' if y == t else 'WRONG'}"
                for x, (y, t) in enumerate(zip(got, tab))]
        with open(os.path.join(outdir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {name}: strict NCT circuit, {len(circ)} gates, QC {quantum_cost(circ)}, "
                    f"no ancilla, verified on all {1 << n} inputs: {ok}\n")
            f.write("# line x0 = most significant bit; gates are applied top to bottom\n")
            f.write("# gate format: 1 = control, 2 = unused, 3 = target\n\n")
            for k, g in enumerate(circ, 1):
                f.write(f"{k:2d}. {gate_str(g):24s} {g}\n")
            f.write("\ntruth table check:\n" + "\n".join(rows) + "\n")
        with open(os.path.join(outdir, f"{name}.tfc"), "w", encoding="utf-8") as f:
            f.write(to_tfc(circ, n, name))
        print("  truth table: " + ("all 16 rows match" if ok else "MISMATCH"))


if __name__ == "__main__":
    main()
