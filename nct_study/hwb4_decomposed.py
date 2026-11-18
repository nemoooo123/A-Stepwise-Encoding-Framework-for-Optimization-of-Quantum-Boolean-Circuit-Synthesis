"""
hwb4 built by decomposition into small (3-line) blocks, Bennett style:

    hwb(x) = rotate_left(x, weight(x))

  lines: x0 x1 x2 x3 (data, x0 = MSB) + w0 w1 (ancilla, start at 0, must end at 0)

  (1) compute   w = weight(x) mod 4  into (w1 w0)   - for each data bit b: w1 ^= b.w0 ; w0 ^= b
                (a 2-bit counter; each step is a 3-line block: 1 Toffoli + 1 CNOT)
  (2) use       rotate left by 1 if w0, by 2 if w1  - controlled swaps (Fredkin, 3-line blocks),
                Fredkin(c; a, b) = CNOT(b->a) TOF(c,a->b) CNOT(b->a)
  (3) uncompute rotation keeps the weight, so recount on the OUTPUT in reverse -> w back to 0

Every block acts on at most 3 lines; the circuit is strict NCT. Verified on all 16 inputs.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_core import gate_str
from nct_convert import simulate, quantum_cost, optimize_nct, WindowTable
from utils.data_loader import DataLoader

N_LINES = 6
X = [0, 1, 2, 3]
W0, W1 = 4, 5


def g(ctrls, t):
    v = [2] * N_LINES
    v[t] = 3
    for c in ctrls:
        v[c] = 1
    return v


def fredkin(c, a, b):
    return [g([b], a), g([c, a], b), g([b], a)]


def count_weight():
    blocks = []
    for b in X:
        blocks.append(("count +x%d" % b, [g([b, W0], W1), g([b], W0)]))
    return blocks


def rotate():
    rot1 = fredkin(W0, 0, 1) + fredkin(W0, 1, 2) + fredkin(W0, 2, 3)   # rotate left by 1 if w0
    rot2 = fredkin(W1, 0, 2) + fredkin(W1, 1, 3)                       # rotate left by 2 if w1
    return [("rotate by 1 (if w0): 3 Fredkin", rot1), ("rotate by 2 (if w1): 2 Fredkin", rot2)]


def main():
    blocks = count_weight() + rotate()
    blocks += [("uncount " + name[6:], list(reversed(gs))) for name, gs in reversed(count_weight())]
    circ = [x for _, gs in blocks for x in gs]

    tab = DataLoader().get_output(4, 1)
    out = simulate(circ, N_LINES)
    ok = all(out[x << 2] == (tab[x] << 2) for x in range(16))   # ancilla in = 0, ancilla out = 0
    print(f"decomposed hwb4: {len(circ)} NCT gates, QC {quantum_cost(circ)}, 2 clean ancilla lines, "
          f"correct on all 16 inputs: {ok}")
    for name, gs in blocks:
        print(f"  [{name}]  " + " ; ".join(gate_str(x).replace('x4', 'w0').replace('x5', 'w1') for x in gs))
    opt = optimize_nct(circ, [WindowTable(3, 20)])
    out2 = simulate(opt, N_LINES)
    ok2 = all(out2[x << 2] == (tab[x] << 2) for x in range(16))
    print(f"after deletion/MR + 3-line windows: {len(opt)} gates, QC {quantum_cost(opt)}, correct: {ok2}")


if __name__ == "__main__":
    main()
