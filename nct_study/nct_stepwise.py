"""
Prototype: stepwise encoding for strict-NCT synthesis (Hamming-distance guided).

Decoding (deterministic given a chromosome):
    r = target function (residual).  For step k = 0 .. L-1, until r is the identity:
      1. candidates = every NCT gate on the circuit lines (the count is constant at every step;
         only the immediately previous gate is excluded because it would undo the last step)
      2. rank candidates by  H(g o r) = sum_x HD(x, (g o r)(x))  (ties: cheaper gate, then index)
      3. gene k (b bits) picks rank  (gene mod #candidates)   -> 0 = greedy best choice
      4. r <- g o r   (the gate is stacked at the output end)
    If r is still not the identity after L steps, a guaranteed constructor finishes it:
    transformation-based synthesis (Miller-Maslov-Dueck basic algorithm, positive-control MCT
    gates) converted to NCT with nct_convert. So EVERY chromosome decodes to a correct circuit.
    Finally the whole circuit gets deletion/MR + 3-line window resynthesis.

Parity: for n >= 4 an odd target cannot be built from NCT on n lines, so one dirty ancilla line
is added from the start (target = f (x) identity on n+1 lines).

Chromosome = [gene_0 | gene_1 | ... | gene_{L-1}], gene_k = b bits (rank among candidates).

Search: AE-QTS as in core/AE_QTS.py - each bit has P(bit = 1); the t-th best and t-th worst
neighbour are compared and every differing bit moves delta/(t+1) toward the better one.

Usage:
    python nct_study/nct_stepwise.py validate     # step 1: greedy (all-zero genes) + correctness
    python nct_study/nct_stepwise.py search       # step 2: AE-QTS on the 3/4-bit catalog problems
"""

import argparse
import collections
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from nct_core import nct_library, gate_perm, parity, with_dirty_ancilla, NCTOracle
from nct_convert import convert_to_nct, optimize_nct, quantum_cost, simulate, WindowTable

_T3 = None


def window_table():
    global _T3
    if _T3 is None:
        _T3 = WindowTable(3, 20)
    return _T3


# ---------------------------------------------------------------------------
# guaranteed completion: basic transformation-based synthesis (output side)
# ---------------------------------------------------------------------------

def mmd_output_side(r, lines):
    """Gates h_1..h_m with h_m o ... o h_1 o r = identity (positive-control MCT gates)."""
    r = np.array(r, dtype=np.int64)
    gates = []

    def apply(target_bit, ctrl_mask):
        g = [2] * lines
        g[lines - 1 - target_bit] = 3
        for j in range(lines):
            if ctrl_mask >> j & 1:
                g[lines - 1 - j] = 1
        sel = (r & ctrl_mask) == ctrl_mask
        r[sel] ^= 1 << target_bit
        gates.append(g)

    for i in range(1 << lines):
        o = int(r[i])
        if o == i:
            continue
        for j in range(lines):                     # bits that must turn 1 -> control on ones(o)
            if i >> j & 1 and not o >> j & 1:
                apply(j, o)
                o = int(r[i])
        for j in range(lines):                     # bits that must turn 0 -> control on ones(i)
            if not i >> j & 1 and o >> j & 1:
                apply(j, i)
                o = int(r[i])
    assert np.array_equal(r, np.arange(1 << lines))
    return gates


# ---------------------------------------------------------------------------
# the encoding
# ---------------------------------------------------------------------------

class StepwiseNCT:
    def __init__(self, target, n, L, b=2):
        tab = np.asarray(target, dtype=np.uint8)
        self.n = n
        self.anc = n >= 4 and parity(tab) == 1
        self.lines = n + 1 if self.anc else n
        self.target = with_dirty_ancilla(tab, n) if self.anc else tab
        self.lib = nct_library(self.lines)
        self.gp = np.stack([gate_perm(g) for g in self.lib]).astype(np.int64)
        self.gqc = np.array([quantum_cost([g]) for g in self.lib])
        N = 1 << self.lines
        self.popc = np.array([bin(i).count("1") for i in range(N)], dtype=np.int64)
        self.ident = np.arange(N, dtype=np.int64)
        self.idx = np.arange(len(self.lib))
        self.L, self.b = L, b
        self.cache = {}

    @property
    def nbits(self):
        return self.L * self.b

    def genes(self, bits):
        bits = np.asarray(bits).reshape(self.L, self.b)
        return (bits * (1 << np.arange(self.b - 1, -1, -1))).sum(1)

    def decode(self, bits):
        """Returns (full circuit in time order, number of encoded steps used, fallback gate count)."""
        r = self.target.astype(np.int64)
        chosen, prev = [], -1
        visited = {r.tobytes()}
        for gene in self.genes(bits):
            if np.array_equal(r, self.ident):
                break
            cand = self.gp[:, r]                                     # (G, N): g o r
            h = self.popc[cand ^ self.ident].sum(1)
            # candidates that return to an already visited residual go to the back of the
            # ranking (not removed, so the candidate count stays constant)
            seen = np.array([row.tobytes() in visited for row in cand])
            order = np.lexsort((self.idx, self.gqc, h, seen))
            if prev >= 0:
                order = order[order != prev]
            g = int(order[int(gene) % len(order)])
            r = cand[g]
            visited.add(r.tobytes())
            chosen.append(g)
            prev = g
        prefix = [list(self.lib[i]) for i in reversed(chosen)]
        if np.array_equal(r, self.ident):
            return prefix, len(chosen), 0, self.lines
        # finish the residual with the guaranteed constructor, converted to NCT
        hs = mmd_output_side(r, self.lines)
        conv = convert_to_nct(list(reversed(hs)), self.lines)
        lines = conv["lines"]
        if lines > self.lines:
            prefix = [g + [2] for g in prefix]
        # f = g_1 o ... o g_L o r_L : the residual's circuit runs first, then the chosen gates
        return conv["nct"] + prefix, len(chosen), len(hs), lines

    def evaluate(self, bits, key="gc"):
        k = np.asarray(bits, dtype=np.uint8).tobytes()
        if k not in self.cache:
            circ, used, fb, lines = self.decode(bits)
            circ = optimize_nct(circ, [window_table()])
            gc, qc = len(circ), quantum_cost(circ)
            extra_lines = lines - self.n
            self.cache[k] = {"gc": gc, "qc": qc, "lines": lines, "extra": extra_lines,
                             "fallback": fb, "circ": circ}
        e = self.cache[k]
        primary = (e["gc"], e["qc"]) if key == "gc" else (e["qc"], e["gc"])
        return (e["extra"],) + primary, e      # never prefer a solution that needs more lines

    def check(self, circ, lines):
        got = simulate(circ, lines)
        want = np.asarray(self.target, dtype=np.int64)
        if lines == self.lines:
            return bool(np.array_equal(got, want))
        exp = np.empty(2 * want.size, dtype=np.int64)
        exp[0::2], exp[1::2] = 2 * want, 2 * want + 1
        return bool(np.array_equal(got, exp))


# ---------------------------------------------------------------------------
# AE-QTS on a plain bit string (same update rule as core/AE_QTS.updateQ)
# ---------------------------------------------------------------------------

def ae_qts(model, pop=10, gens=1000, delta=0.005, seed=0, key="gc"):
    rng = np.random.default_rng(seed)
    p1 = np.full(model.nbits, 0.5)
    best_fit, best = None, None
    for _ in range(gens):
        X = (rng.random((pop, model.nbits)) < p1).astype(np.uint8)
        res = [model.evaluate(x, key) for x in X]
        order = sorted(range(pop), key=lambda i: res[i][0])
        if best_fit is None or res[order[0]][0] < best_fit:
            best_fit, best = res[order[0]][0], res[order[0]][1]
        for t in range(pop // 2):
            bx, wx = X[order[t]], X[order[pop - 1 - t]]
            diff = bx != wx
            step = delta / (t + 1)
            p1[diff] += np.where(bx[diff] == 1, step, -step)
        np.clip(p1, 0.0, 1.0, out=p1)
    return best


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def default_L(n):
    return {3: 20, 4: 40}.get(n, 15 * n)


def validate():
    from utils.data_loader import DataLoader
    import itertools
    t0 = time.time()
    print("== Step 1: greedy decoding (all genes = 0) + guaranteed completion ==")
    # all 3-bit functions vs the exact optimum
    o3 = NCTOracle(3, 20, verbose=False)
    dist_g, dist_o = collections.Counter(), collections.Counter()
    fb_used = bad = 0
    for p in itertools.permutations(range(8)):
        m = StepwiseNCT(p, 3, default_L(3))
        _, e = m.evaluate(np.zeros(m.nbits, dtype=np.uint8))
        if not m.check(e["circ"], e["lines"]):
            bad += 1
        fb_used += e["fallback"] > 0
        dist_g[e["gc"]] += 1
        dist_o[o3.lookup(np.array(p, dtype=np.uint8))] += 1
    ks = sorted(set(dist_g) | set(dist_o))
    print(f"3-bit, all 40320 functions: incorrect = {bad}, needed completion = {fb_used}")
    print("gates    : " + " ".join(f"{k:>6}" for k in ks))
    print("optimal  : " + " ".join(f"{dist_o.get(k, 0):>6}" for k in ks))
    print("greedy   : " + " ".join(f"{dist_g.get(k, 0):>6}" for k in ks))
    avg_o = sum(k * v for k, v in dist_o.items()) / 40320
    avg_g = sum(k * v for k, v in dist_g.items()) / 40320
    print(f"average gates: optimal {avg_o:.3f}, greedy {avg_g:.3f}")

    rng = np.random.default_rng(1)
    for n, count in ((4, 300), (5, 60)):
        bad = fb = extra = 0
        for _ in range(count):
            p = rng.permutation(1 << n)
            m = StepwiseNCT(p, n, default_L(n))
            _, e = m.evaluate(np.zeros(m.nbits, dtype=np.uint8))
            bad += not m.check(e["circ"], e["lines"])
            fb += e["fallback"] > 0
            extra += e["extra"] > (1 if m.anc else 0)
        print(f"{n}-bit, {count} random permutations: incorrect = {bad}, needed completion = {fb}, "
              f"needed an extra line beyond parity = {extra}")

    loader = DataLoader()
    print("\ncatalog (greedy):  name  lines  gates  QC  completion-gates")
    for n in (3, 4):
        for idx in sorted(loader.data_map[n]):
            m = StepwiseNCT(loader.get_output(n, idx), n, default_L(n))
            _, e = m.evaluate(np.zeros(m.nbits, dtype=np.uint8))
            assert m.check(e["circ"], e["lines"])
            print(f"  {loader.names[n][idx]:16s} {e['lines']}  {e['gc']:4d} {e['qc']:5d}  {e['fallback']}")
    print(f"[{time.time() - t0:.0f}s]")


def _search_one(args):
    n, idx, runs, pop, gens, delta, b, key = args
    from utils.data_loader import DataLoader
    loader = DataLoader()
    m = StepwiseNCT(loader.get_output(n, idx), n, default_L(n), b=b)
    _, greedy = m.evaluate(np.zeros(m.nbits, dtype=np.uint8), key)
    results = []
    t0 = time.time()
    for s in range(runs):
        e = ae_qts(m, pop=pop, gens=gens, delta=delta, seed=s, key=key)
        assert m.check(e["circ"], e["lines"])
        results.append(e)
    return (loader.names[n][idx], n, m.lines, greedy, results, time.time() - t0)


def search(runs, pop, gens, delta, b, key):
    jobs = [(3, 1), (3, 2)] + [(4, i) for i in range(1, 9)]
    opt = {"f3_1": (5, 13), "nth_prime3_inc": (4, 8), "hwb4": (11, 23), "4_49": (12, "-"),
           "nth_prime4_inc": (12, "-")}
    beam = {"hwb4": (12, 28), "4_49": (12, 32), "nth_prime4_inc": (13, 45),
            "f3_1": (5, 13), "nth_prime3_inc": (4, 8)}
    saeedi = {"nth_prime3_inc": 6, "hwb4": 23, "4_49": 32, "nth_prime4_inc": 51}
    conv = {"f3_1": (5, 13), "nth_prime3_inc": (4, 8), "hwb4": (92, 392), "4_49": (68, 268),
            "4b15g_1": (66, 254), "4b15g_2": (90, 366), "4b15g_3": (67, 279), "4b15g_4": (87, 367),
            "4b15g_5": (76, 312), "nth_prime4_inc": (61, 257)}
    out = [f"# NCT 步驟式編碼原型：AE-QTS 搜尋結果", "",
           f"設定：族群 {pop}、世代 {gens}、Δ = {delta}、每個基因 {b} 位元、每題 {runs} 次、評分 = {key}。",
           "每格為 `閘數 / QC`；所有電路都模擬驗證正確。", "",
           "| 題目 | 線數 | 貪婪 (基因全 0) | AE-QTS 最佳 | AE-QTS 平均閘數 | beam search | MPMCT→NCT 轉換 | NCT 最佳 | Saeedi QC |",
           "|---|---|---|---|---|---|---|---|---|"]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as ex:
        for name, n, lines, greedy, res, secs in ex.map(
                _search_one, [(n, i, runs, pop, gens, delta, b, key) for n, i in jobs]):
            best = min(res, key=lambda e: (e["gc"], e["qc"]))
            avg = np.mean([e["gc"] for e in res])
            row = (f"| {name} | {lines} | {greedy['gc']} / {greedy['qc']} | **{best['gc']} / {best['qc']}** | "
                   f"{avg:.1f} | {'%s / %s' % beam[name] if name in beam else '-'} | "
                   f"{'%s / %s' % conv[name] if name in conv else '-'} | "
                   f"{'%s / %s' % opt[name] if name in opt else '-'} | {saeedi.get(name, '-')} |")
            print(row + f"   [{secs:.0f}s]", flush=True)
            out.append(row)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "stepwise_nct_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "search"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--bits-per-gene", type=int, default=2)
    ap.add_argument("--fitness", choices=["gc", "qc"], default="gc")
    a = ap.parse_args()
    if a.mode == "validate":
        validate()
    else:
        search(a.runs, a.pop, a.gens, a.delta, a.bits_per_gene, a.fitness)


if __name__ == "__main__":
    main()
