"""
Exhaustive experiments: how do full-control MPMCT gates (the gates this framework emits)
convert to NCT, alone and together?

  E1  n=3  every single MPMCT gate      -> exact optimal NCT cost
  E2  n=3  every ordered pair of gates   -> cost(A then B) vs cost(A) + cost(B)
  E3  n=4  every pair (up to line relabeling) -> exact optimal NCT cost (no ancilla)
  E4  all catalog problems              -> permutation parity (is NCT possible without ancilla?)
  E5  circuits actually produced by AE-QTS (n=3, n=4):
        - per-gate conversion vs optimal block-wise conversion (DP over consecutive blocks)
        - n=4 with one dirty ancilla: rule-based Barenco conversion, fixed vs greedy parameters

Run from anywhere:  python nct_study/nct_experiments.py [--quick]
Results are printed and written to nct_study/results/report.md
"""

import os
import sys
import time
import copy
import itertools
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from nct_core import (NCTOracle, gate_perm, circuit_perm, gate_str, nct_library,
                      full_control_gates, parity, with_dirty_ancilla)

QUICK = "--quick" in sys.argv
OUT = []


def emit(line=""):
    print(line, flush=True)
    OUT.append(line)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def n_neg(g):
    return sum(1 for v in g if v == 0)


def edge_vertices(g):
    n = len(g)
    t = g.index(3)
    base = sum(v << (n - 1 - i) for i, v in enumerate(g) if v in (0, 1))
    return base, base | (1 << (n - 1 - t))


def relation(a, b):
    """Geometric relation of two hypercube edges."""
    ta, tb = a.index(3), b.index(3)
    d = min(bin(x ^ y).count("1") for x in edge_vertices(a) for y in edge_vertices(b))
    if ta == tb:
        return f"同目標 (平行邊), 控制差 {d} 位"
    return "不同目標, 共用頂點" if d == 0 else f"不同目標, 端點距離 {d}"


def synth(oracle, perm):
    """Recover one optimal NCT circuit for a permutation inside the BFS ball (gate descent)."""
    perm = np.asarray(perm, dtype=np.uint8)
    d = oracle.lookup(perm)
    if d is None:
        return None
    lib = nct_library(oracle.n)
    out = []
    while d > 0:
        for g, gp in zip(lib, oracle.lib):
            q = gp[perm]  # strip g off the end: perm = g o q
            if oracle.lookup(q) == d - 1:
                out.append(g)
                perm, d = q, d - 1
                break
    return out[::-1]


def commute(g, h):
    tg, th = g.index(3), h.index(3)
    if tg == th:
        return True
    for i in range(len(g)):  # conflicting polarities on a shared control: never fire together
        if i not in (tg, th) and g[i] in (0, 1) and h[i] in (0, 1) and g[i] != h[i]:
            return True
    return h[tg] == 2 and g[th] == 2


def peephole(gates):
    """A*A = I with the moving rule: each incoming gate slides left past commuting gates
    and cancels against the first identical one it meets."""
    out = []
    for g in gates:
        j = len(out) - 1
        while j >= 0:
            if out[j] == g:
                del out[j]
                break
            if not commute(out[j], g):
                j = -1
                break
            j -= 1
        if j < 0:
            out.append(g)
    return out


def rule_mpmct_to_toffoli_n3(g):
    """Textbook rule for a 2-control MPMCT on 3 lines: NOTs around the negative controls."""
    nots = [[3 if j == i else 2 for j in range(3)] for i, v in enumerate(g) if v == 0]
    tof = [1 if v in (0, 1) else 3 for v in g]
    return nots + [tof] + nots


def barenco_with_ancilla(g, split, order):
    """
    Full-control 3-control MPMCT on 4 lines -> NCT on 5 lines (line 4 = dirty ancilla a).
      split: which control stays with the target Toffoli (the other two feed the ancilla)
      order: 0 -> P G P G, 1 -> G P G P,  P = TOF(p,q -> a), G = TOF(a,r -> t)
    Negative controls are handled with NOT pairs around the whole block.
    """
    t = g.index(3)
    ctrls = [i for i in range(4) if i != t]
    r = ctrls[split]
    p, q = [c for c in ctrls if c != r]
    a = 4

    def mk(cs, tgt):
        x = [2] * 5
        x[tgt] = 3
        for c in cs:
            x[c] = 1
        return x

    P, G = mk([p, q], a), mk([a, r], t)
    body = [P, G, P, G] if order == 0 else [G, P, G, P]
    nots = [[3 if j == i else 2 for j in range(5)] for i in ctrls if g[i] == 0]
    return nots + body + nots


def table(headers, rows):
    emit("| " + " | ".join(headers) + " |")
    emit("|" + "---|" * len(headers))
    for r in rows:
        emit("| " + " | ".join(str(x) for x in r) + " |")
    emit()


# ---------------------------------------------------------------------------
# E1 / E2  (n = 3, exact over the whole group)
# ---------------------------------------------------------------------------

def run_n3(o3):
    gates = full_control_gates(3)
    emit("## E1  n=3 單一 MPMCT 閘的最佳 NCT 成本")
    emit()
    single = {tuple(g): o3.lookup(gate_perm(g)) for g in gates}
    by_neg = collections.defaultdict(list)
    for g in gates:
        by_neg[n_neg(g)].append(single[tuple(g)])
    rows = []
    for k in sorted(by_neg):
        rows.append((k, len(by_neg[k]), sorted(set(by_neg[k])), 1 + 2 * k))
    table(["負控制數", "閘數", "最佳 NCT 成本", "規則轉換 (NOT 包夾) 成本"], rows)
    for g in gates:
        if n_neg(g) == 2:
            emit(f"例: {gate_str(g)} 最佳 = " + " ; ".join(gate_str(x) for x in synth(o3, gate_perm(g))))
            break
    emit()

    emit("## E2  n=3 兩個 MPMCT 閘一起轉換 (所有 132 個有序閘對)")
    emit()
    emit("節省 = cost(A)+cost(B) − cost(A·B)，全部為精確最佳值。")
    emit()
    agg = collections.defaultdict(list)
    agg_neg = collections.defaultdict(list)
    examples = {}
    for a, b in itertools.permutations(gates, 2):
        c = o3.lookup(gate_perm(b)[gate_perm(a)])
        s = single[tuple(a)] + single[tuple(b)]
        rel = relation(a, b)
        agg[rel].append((s, c))
        agg_neg[(rel, n_neg(a) + n_neg(b))].append(s - c)
        if rel not in examples or s - c > examples[rel][0]:
            examples[rel] = (s - c, a, b, c)
    rows = []
    for rel, v in sorted(agg.items()):
        sav = [s - c for s, c in v]
        rows.append((rel, len(v), f"{np.mean([s for s, _ in v]):.2f}", f"{np.mean([c for _, c in v]):.2f}",
                     f"{np.mean(sav):.2f}", min(sav), max(sav)))
    table(["關係", "對數", "分開轉 平均", "一起轉 平均", "平均節省", "最小節省", "最大節省"], rows)

    emit("依負控制總數細分 (平均節省):")
    emit()
    negs = sorted({k for _, k in agg_neg})
    rows = []
    for rel in sorted(agg):
        rows.append([rel] + [f"{np.mean(agg_neg[(rel, k)]):.2f}" if (rel, k) in agg_neg else "-" for k in negs])
    table(["關係"] + [f"負控制={k}" for k in negs], rows)

    emit("各關係中節省最多的例子 (一起轉換的最佳 NCT 電路):")
    emit()
    for rel, (sav, a, b, c) in sorted(examples.items()):
        circ = synth(o3, gate_perm(b)[gate_perm(a)])
        emit(f"- **{rel}**: {gate_str(a)} ; {gate_str(b)} → {c} 閘 (省 {sav}): "
             + " ; ".join(gate_str(x) for x in circ))
    emit()


# ---------------------------------------------------------------------------
# E3  (n = 4, meet in the middle, exact up to cost 12)
# ---------------------------------------------------------------------------

def canon_pair(a, b):
    best = None
    for sigma in itertools.permutations(range(4)):
        ra = [None] * 4
        rb = [None] * 4
        for i in range(4):
            ra[sigma[i]] = a[i]
            rb[sigma[i]] = b[i]
        k = min((tuple(ra), tuple(rb)), (tuple(rb), tuple(ra)))
        best = k if best is None or k < best else best
    return best


def run_n4_pairs(o4):
    emit("## E3  n=4 兩個 MPMCT 閘一起轉換 (無輔助位元)")
    emit()
    emit("n=4 時每個 full-control MPMCT 閘都是一個 **奇置換** (只交換兩個狀態)，而 NOT/CNOT/Toffoli 在 4 條線上都是偶置換，"
         "所以 **單一閘無法在不加線的情況下轉成 NCT**，至少要兩個閘湊成偶置換才轉得出來。")
    emit()
    gates = full_control_gates(4)
    classes = collections.defaultdict(list)
    for a, b in itertools.permutations(gates, 2):
        classes[canon_pair(a, b)].append((a, b))
    emit(f"有序閘對 {32 * 31} 個，依線重新標號與反轉 (A·B 與 B·A 互為反函數、成本相同) 化簡後有 {len(classes)} 類。")
    emit()
    t0 = time.time()
    agg = collections.defaultdict(list)
    detail = []
    for key, members in classes.items():
        a, b = [list(x) for x in key]
        c = o4.cost(gate_perm(b)[gate_perm(a)])
        rel = relation(a, b)
        agg[rel].append((c, len(members), n_neg(a) + n_neg(b)))
        detail.append((rel, n_neg(a) + n_neg(b), gate_str(a), gate_str(b), c if c is not None else ">12"))
    emit(f"(計算時間 {time.time() - t0:.0f}s)")
    emit()
    rows = []
    for rel, v in sorted(agg.items()):
        costs = [c for c, _, _ in v if c is not None]
        w = [m for c, m, _ in v if c is not None]
        unk = sum(1 for c, _, _ in v if c is None)
        rows.append((rel, len(v), min(costs) if costs else "-", max(costs) if costs else "-",
                     f"{np.average(costs, weights=w):.2f}" if costs else "-", unk))
    table(["關係", "類別數", "最佳成本 min", "最佳成本 max", "加權平均", ">12 的類別"], rows)
    emit("全部類別明細:")
    emit()
    table(["關係", "負控制總數", "A", "B", "一起轉 最佳 NCT"], sorted(detail, key=lambda r: (r[0], r[1], str(r[4]))))
    anc = [4 + 2 * n_neg(g) for g in gates]
    emit(f"對照：加一條 dirty ancilla 時，單一閘可用 Barenco 拆成 4 個 Toffoli + 2×(負控制數) 個 NOT "
         f"(成本 {min(anc)}~{max(anc)})，所以分開轉換一對閘的成本是 8~20。")
    emit()


# ---------------------------------------------------------------------------
# E4 parity of every catalog problem
# ---------------------------------------------------------------------------

def run_parity():
    from utils.data_loader import DataLoader
    emit("## E4  題庫中每個函數的奇偶性")
    emit()
    emit("n≥4 時 NCT (不加線) 只能實現偶置換。奇置換的題目 **一定需要至少一條輔助線**，不論用什麼轉換方法。")
    emit()
    loader = DataLoader()
    rows = []
    for bits in sorted(loader.data_map):
        for idx in sorted(loader.data_map[bits]):
            tab = loader.data_map[bits][idx]
            p = parity(tab)
            need = "否 (n=3 可實現全部置換)" if bits <= 3 else ("**需要**" if p else "不需要")
            rows.append((bits, idx, loader.names[bits][idx], "奇" if p else "偶", need))
    table(["n", "idx", "名稱", "奇偶", "NCT 需要輔助線?"], rows)


# ---------------------------------------------------------------------------
# E5 real circuits from AE-QTS
# ---------------------------------------------------------------------------

def mpmct_circuit(bits, idx, iters, seed):
    import random
    from utils.data_loader import DataLoader
    from utils.init_state import find_cycles, build_encode
    from core.AE_QTS import AE_QTS_run_single_experiment
    random.seed(seed)
    np.random.seed(seed)
    tab = DataLoader().get_output(bits, idx)
    cycles, needed = find_cycles(tab, check_zero_gate=True)
    q1, q2, q3, q4, enc, traj = build_encode(cycles)
    hist = np.full((1, iters), float("inf"))
    _, cnt, circ = AE_QTS_run_single_experiment(
        max_iterations=iters, rotation_cycles=cycles, num_neighbors=50, num_bits=bits,
        base_trajectory=traj, experiment_id=0, encoding_table=enc,
        qindividuals1=copy.deepcopy(q1), qindividuals2=copy.deepcopy(q2),
        qindividuals3=copy.deepcopy(q3), qindividuals4=copy.deepcopy(q4),
        fitness_history_matrix=hist, target_output=tab, delta_theta=0.005)
    perm = circuit_perm(circ, bits)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=np.uint8)
    ok = list(perm) == tab or list(inv) == tab
    return tab, [list(g) for g in circ], ok


def block_dp(gates, cost_fn, max_len, step):
    """Optimal split of the gate sequence into consecutive blocks, each converted optimally."""
    L = len(gates)
    perms = [gate_perm(g) for g in gates]
    INF = 10 ** 9
    best = [0] + [INF] * L
    back = [0] * (L + 1)
    cache = {}
    for j in range(1, L + 1):
        for ln in range(step, min(max_len, j) + 1, step):
            i = j - ln
            if best[i] >= INF:
                continue
            p = perms[i]
            for q in perms[i + 1:j]:
                p = q[p]
            k = p.tobytes()
            if k not in cache:
                cache[k] = cost_fn(p)
            c = cache[k]
            if c is not None and best[i] + c < best[j]:
                best[j], back[j] = best[i] + c, i
    if best[L] >= INF:
        return None, None
    segs, j = [], L
    while j > 0:
        segs.append(j - back[j])
        j = back[j]
    return best[L], segs[::-1]


def run_real(o3, o4):
    iters = 60 if QUICK else 300
    emit("## E5  框架實際產生的 MPMCT 電路")
    emit()
    emit(f"用 AE-QTS ({iters} 次迭代、50 個鄰居、seed 固定) 產生電路，再比較不同的 NCT 轉換方式。")
    emit()

    emit("### n=3")
    emit()
    rows = []
    for idx in (1, 2):
        tab, circ, ok = mpmct_circuit(3, idx, iters, seed=idx)
        per_gate_opt = sum(o3.lookup(gate_perm(g)) for g in circ)
        rule = []
        for g in circ:
            rule += rule_mpmct_to_toffoli_n3(g)
        rule_pp = peephole(rule)
        assert list(circuit_perm(rule_pp, 3)) == list(circuit_perm(circ, 3))
        pair, _ = block_dp(circ, o3.lookup, 2, 1)
        dp, segs = block_dp(circ, o3.lookup, len(circ), 1)
        whole = o3.lookup(circuit_perm(circ, 3))
        rows.append((f"3_{idx}", len(circ), "✓" if ok else "✗", len(rule), len(rule_pp), per_gate_opt,
                     pair, dp, whole))
    table(["題目", "MPMCT 閘數", "驗證", "規則轉換", "規則+peephole", "逐閘最佳",
           "區塊≤2 DP", "任意區塊 DP", "整個函數最佳"], rows)

    emit("### n=4")
    emit()
    rows = []
    rows_anc = []
    probs = (1, 2, 8) if QUICK else tuple(range(1, 9))
    for idx in probs:
        tab, circ, ok = mpmct_circuit(4, idx, iters, seed=idx)
        par = parity(tab)
        # (a) with one dirty ancilla: rule-based Barenco, fixed parameters vs greedy choice
        fixed = []
        for g in circ:
            fixed += barenco_with_ancilla(g, 0, 0)
        fixed_pp = peephole(fixed)
        greedy = []
        for g in circ:
            best = None
            for split in range(3):
                for order in range(2):
                    cand = peephole(greedy + barenco_with_ancilla(g, split, order))
                    if best is None or len(cand) < len(best):
                        best = cand
            greedy = best
        target5 = with_dirty_ancilla(circuit_perm(circ, 4), 4)
        assert list(circuit_perm(fixed_pp, 5)) == list(target5)
        assert list(circuit_perm(greedy, 5)) == list(target5)
        rows_anc.append((f"4_{idx}", len(circ), len(fixed), len(fixed_pp), len(greedy)))
        # (b) no ancilla: blocks must have even length; each block cost exact up to 11
        if par:
            rows.append((f"4_{idx}", len(circ), "奇", "不可能 (需輔助線)", "-", "-"))
            continue
        pair, _ = block_dp(circ, lambda p: o4.cost(p), 2, 2)
        dp, segs = block_dp(circ, lambda p: o4.cost(p), 4 if QUICK else 6, 2)
        seg_hist = dict(collections.Counter(segs)) if segs else "-"
        rows.append((f"4_{idx}", len(circ), "偶", pair if pair is not None else "無解", dp, seg_hist))
        print(f"  4_{idx} done", flush=True)
    emit("**無輔助線** (每個區塊必須是偶數個閘；區塊成本為精確最佳值，≤11)")
    emit()
    table(["題目", "MPMCT 閘數", "奇偶", "兩兩配對 DP", f"區塊≤{4 if QUICK else 6} DP", "區塊長度分布"], rows)
    emit("**加一條 dirty ancilla** (規則式 Barenco 分解 + peephole；已模擬驗證正確)")
    emit()
    table(["題目", "MPMCT 閘數", "固定參數 (未化簡)", "固定參數 + peephole", "每閘貪婪選參數 + peephole"], rows_anc)


def main():
    t0 = time.time()
    emit("# MPMCT → NCT 窮舉實驗報告")
    emit()
    o3 = NCTOracle(3, 20, verbose=False)
    run_n3(o3)
    run_parity()
    print("building n=4 BFS ball (depth 6)...", flush=True)
    o4 = NCTOracle(4, 6)
    run_n4_pairs(o4)
    o4.left_depth = 5  # blocks in E5: exact up to 5+6 = 11, much faster per query
    run_real(o3, o4)
    emit(f"_總執行時間 {time.time() - t0:.0f}s_")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
