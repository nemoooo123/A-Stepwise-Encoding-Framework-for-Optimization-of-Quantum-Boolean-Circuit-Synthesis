"""
窮舉四層編碼，求出這個編碼框架真正能達到的最小閘數，
用來驗證 AE-QTS / DE 是不是真的找得到最佳解。

做法就是把 AE-QTS 拆掉隨機性：
  * 移除 updateQ（沒有量子態更新）
  * 把 gen_nbrs 的機率取樣換成 for 迴圈窮舉四層編碼
  * decode_and_synthesize / synthesize_route 完全沿用原本的程式碼，不改一行

兩種模式
--------
raw        逐位元窮舉四層編碼的每個 0/1。最忠實，但空間 = 2^(總 bit 數)：
           3-bit 第 1/3/5 題是 2^46 / 2^32 / 2^43，跑不完。
           注意 raw 不套 repair_sequence_logic（那是隨機的，套了就不叫窮舉），
           所以 raw 會多探索 AE-QTS/DE 根本取樣不到的「不平衡 L4」解。

canonical  只窮舉真正會改變電路的自由度，可達電路集合與 raw（限平衡 L4）相同：
             L1 priority_weights 只經過 sorted(reverse=True) 決定 cycle 順序
                 -> C! 種，而不是 2^(C*b1)
             L2 entry_points 會做 % node_mapping_index
                 -> limit 種，而不是 2^b2
             L3 mid_node_matrix 同樣只決定 bit-flip 的先後
                 -> 每個 step d! 種，而不是 2^(d*ceil(log2 d))
             L4 operation_sequences 經 repair_sequence_logic 強制 0/1 等量
                 -> 長度 2d-2 的平衡向量 C(2d-2, d-1) 種，而不是 2^(2d-2)
             每個 cycle 只走 total_steps-1 步，entry 跳過的那個 step
                 其 L3/L4 不影響電路 -> 不窮舉

--verify   在兩種模式都跑得動的題目上，比對 canonical 與「raw 限平衡 L4」的
           最小值和可達閘數集合，確認縮減沒有漏解。
"""
import argparse
from collections import Counter
from itertools import combinations, permutations, product
from math import comb, factorial

from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode
from utils.topology import decode_and_synthesize

BATCH = 4096


# ---------- 問題結構 ----------

def describe(target):
    """把 build_encode 的輸出整理成窮舉需要的欄位描述。"""
    cycles, _ = find_cycles(target, check_zero_gate=True)
    if not cycles:
        # 恆等映射：不需要任何閘，build_encode 會在 log2(0) 掛掉，直接短路。
        return {"n_cycles": 0, "table": [], "traj": [], "l1_bits": [],
                "l2_bits": [], "limits": [], "cycles": []}
    q1, q2, q3, q4, table, traj = build_encode(cycles)

    n_cycles = len(q1)
    info = {
        "n_cycles": n_cycles,
        "table": table,
        "traj": traj,
        "l1_bits": [len(m) for m in q1],
        "l2_bits": [len(m) for m in q2],
        "limits": list(table),
        "cycles": [],
    }
    for c in range(n_cycles):
        steps = []
        for s in range(len(q3[c])):
            d = len(q3[c][s])                      # sentinel 時為 1
            if d == 1:
                steps.append({"d": 1, "b3": 0, "b4": 0})
            else:
                steps.append({"d": d,
                              "b3": len(q3[c][s][0]),    # 每個 sub 的 bit 數
                              "b4": len(q4[c][s])})      # 2d-2
        total = len(q3[c])
        info["cycles"].append({"steps": steps,
                               "n_steps": total,
                               "total_adj": 2 if total == 1 else total})
    return info


def used_steps(entry, cyc):
    """synthesize_route 實際走訪的 step 索引（與原程式的環狀邏輯一致）。"""
    out, ptr, total = [], entry, cyc["total_adj"]
    for _ in range(total - 1):
        if ptr >= total:
            ptr = 0
        out.append(min(ptr, cyc["n_steps"] - 1))
        ptr += 1
    return out


def bits_of(value, width):
    return [int(b) for b in bin(value)[2:].zfill(width)] if width else []


# ---------- 空間大小 ----------

def space_size(info, mode):
    n = info["n_cycles"]
    if mode == "raw":
        size = 2 ** sum(info["l1_bits"])
        for c, cyc in enumerate(info["cycles"]):
            per_entry = 0
            for e in range(2 ** info["l2_bits"][c]):
                eff = e % info["limits"][c] if e >= info["limits"][c] else e
                prod = 1
                for s in used_steps(eff, cyc):
                    st = cyc["steps"][s]
                    if st["d"] > 1:
                        prod *= 2 ** (st["d"] * st["b3"]) * 2 ** st["b4"]
                per_entry += prod
            size *= per_entry
        return size

    size = factorial(n)
    for c, cyc in enumerate(info["cycles"]):
        per_entry = 0
        for e in range(info["limits"][c]):
            prod = 1
            for s in used_steps(e, cyc):
                st = cyc["steps"][s]
                if st["d"] > 1:
                    prod *= factorial(st["d"]) * comb(st["b4"], st["b4"] // 2)
            per_entry += prod
        size *= per_entry
    return size


# ---------- 基因型組裝 ----------

def make_l1_canonical(order, info):
    """order = cycle 處理順序（優先度高的在前）。回傳 L1 bit 矩陣。"""
    n = info["n_cycles"]
    weight = [0] * n
    for k, c in enumerate(order):
        weight[c] = n - 1 - k        # 遞減權重 -> sorted(reverse=True) 得到 order
    return [bits_of(weight[c], info["l1_bits"][c]) for c in range(n)]


def make_l3_canonical(perm, st):
    """perm = bit-flip 的先後順序（sub 索引）。回傳該 step 的 L3 bit 向量列表。"""
    d = st["d"]
    weight = [0] * d
    for k, sub in enumerate(perm):
        weight[sub] = d - 1 - k
    return [bits_of(weight[i], st["b3"]) for i in range(d)]


def balanced_vectors(length):
    """長度 length、0 與 1 等量的所有 bit 向量（repair_sequence_logic 的值域）。"""
    half = length // 2
    out = []
    for ones in combinations(range(length), half):
        v = [0] * length
        for i in ones:
            v[i] = 1
        out.append(v)
    return out


# ---------- 窮舉主體 ----------

def enumerate_genomes(info, mode, l4_filter=None):
    """產生 (l1, l2, l3, l4) 基因型；格式與 gen_nbrs 的輸出完全相同。"""
    n = info["n_cycles"]
    cycles = info["cycles"]

    if mode == "canonical":
        l1_choices = [make_l1_canonical(o, info) for o in permutations(range(n))]
        l2_ranges = [range(info["limits"][c]) for c in range(n)]
    else:
        ranges = [range(2 ** b) for b in info["l1_bits"]]
        l1_choices = [[bits_of(v, info["l1_bits"][c]) for c, v in enumerate(vals)]
                      for vals in product(*ranges)]
        l2_ranges = [range(2 ** info["l2_bits"][c]) for c in range(n)]

    for l1 in l1_choices:
        for raw_entries in product(*l2_ranges):
            l2 = [bits_of(raw_entries[c], info["l2_bits"][c]) for c in range(n)]
            eff = [raw_entries[c] % info["limits"][c]
                   if raw_entries[c] >= info["limits"][c] else raw_entries[c]
                   for c in range(n)]

            # 逐 cycle 收集「會用到的 step」要窮舉的欄位
            slots, layout = [], []
            for c in range(n):
                for s in used_steps(eff[c], cycles[c]):
                    st = cycles[c]["steps"][s]
                    if st["d"] == 1:
                        continue
                    if mode == "canonical":
                        slots.append([make_l3_canonical(p, st)
                                      for p in permutations(range(st["d"]))])
                        slots.append(balanced_vectors(st["b4"]))
                    else:
                        slots.append([[bits_of(v, st["b3"]) for v in vals]
                                      for vals in product(range(2 ** st["b3"]),
                                                          repeat=st["d"])])
                        vecs = [bits_of(v, st["b4"]) for v in range(2 ** st["b4"])]
                        if l4_filter == "balanced":
                            half = st["b4"] // 2
                            vecs = [v for v in vecs if sum(v) == half]
                        slots.append(vecs)
                    layout.append((c, s))

            for combo in product(*slots):
                l3 = [[[999] for _ in cycles[c]["steps"]] for c in range(n)]
                l4 = [[[0] for _ in cycles[c]["steps"]] for c in range(n)]
                for k, (c, s) in enumerate(layout):
                    l3[c][s] = combo[2 * k]
                    l4[c][s] = combo[2 * k + 1]
                yield l1, l2, l3, l4


def run(target, num_bits, info, mode, l4_filter=None, report_every=1_000_000):
    """跑完窮舉，回傳 (最小閘數, 閘數分佈, 最佳基因型, 已評估數)。"""
    table, traj = info["table"], info["traj"]
    hist = Counter()
    state = {"best": None, "genome": None, "seen": 0, "invalid": 0}
    batch = []

    def record(g, gc):
        hist[gc] += 1
        if state["best"] is None or gc < state["best"]:
            state["best"], state["genome"] = gc, g
        state["seen"] += 1

    def flush():
        if not batch:
            return
        try:
            sols = decode_and_synthesize(
                [g[0] for g in batch], [g[1] for g in batch],
                [g[2] for g in batch], [g[3] for g in batch],
                table, num_bits, len(batch), traj, 1)
            for g, sol in zip(batch, sols):
                record(g, sol[1])
        except (IndexError, ValueError):
            # 不平衡的 L4 會讓 assemble_reversible_circuit 的指標走出軌跡範圍，
            # 這種基因型是結構上無效的（repair_sequence_logic 就是為了擋掉它）。
            # 整批失敗時退回逐個評估，把無效的挑出來計數。
            for g in batch:
                try:
                    sol = decode_and_synthesize([g[0]], [g[1]], [g[2]], [g[3]],
                                                table, num_bits, 1, traj, 1)[0]
                except (IndexError, ValueError):
                    state["invalid"] += 1
                else:
                    record(g, sol[1])
        batch.clear()

    for genome in enumerate_genomes(info, mode, l4_filter):
        batch.append(genome)
        if len(batch) >= BATCH:
            flush()
            if state["seen"] % report_every < BATCH:
                print(f"      ...已評估 {state['seen']:,}，"
                      f"目前最小 {state['best']}", flush=True)
    flush()
    return (state["best"], hist, state["genome"],
            state["seen"], state["invalid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--problem", type=int, default=0, help="0 = 全部")
    ap.add_argument("--mode", choices=["canonical", "raw"], default="canonical")
    ap.add_argument("--verify", action="store_true",
                    help="另外跑 raw（限平衡 L4）與 canonical 對照")
    ap.add_argument("--max-evals", type=float, default=2e8,
                    help="空間大於此值就跳過，避免跑不完")
    args = ap.parse_args()

    loader = DataLoader()
    probs = ([args.problem] if args.problem
             else range(1, len(loader.data_map[args.bits]) + 1))

    summary = []
    for p in probs:
        target = loader.get_output(args.bits, p)
        info = describe(target)
        n_can = space_size(info, "canonical")
        n_raw = space_size(info, "raw")
        lens = [len(t) - 1 for t in info["traj"]]

        print(f"\n{'=' * 72}")
        print(f"{args.bits}-bit 第 {p} 題   cycle 數 {info['n_cycles']}，長度 {lens}")
        print(f"  canonical 空間 {n_can:,}     raw 空間 {n_raw:,}")
        print(f"{'=' * 72}", flush=True)

        size = n_can if args.mode == "canonical" else n_raw
        if size > args.max_evals:
            print(f"  [跳過] {args.mode} 空間 {size:,} 超過 --max-evals {args.max_evals:,.0f}")
            continue

        best, hist, genome, seen, bad = run(target, args.bits, info, args.mode)
        print(f"  {args.mode}: 有效解 {seen:,} 個"
              + (f"，結構無效 {bad:,} 個" if bad else "")
              + f"，最小閘數 = {best}")
        print(f"  閘數分佈（前 8 種）: {sorted(hist.items())[:8]}")
        print(f"  達到最小值的解: {hist[best]:,} 個（{hist[best] / seen * 100:.4f}%）")
        summary.append((p, best, hist[best], seen, hist[best] / seen))

        if args.verify and n_raw <= args.max_evals:
            b2, h2, _, s2, bad2 = run(target, args.bits, info, "raw", l4_filter="balanced")
            print(f"  [verify] raw(限平衡 L4): 有效 {s2:,} / 無效 {bad2:,}，最小 {b2}"
                  f"  -> 最小值一致 {b2 == best}，可達閘數集合一致 {set(hist) == set(h2)}")
            b3, _, _, s3, bad3 = run(target, args.bits, info, "raw")
            print(f"  [verify] raw(含不平衡 L4): 有效 {s3:,} / 無效 {bad3:,}，最小 {b3}"
                  f"  -> repair 擋掉了更好的解: {b3 < best}")

    if summary:
        print(f"\n{'=' * 72}\n總結（{args.bits}-bit, mode={args.mode}）\n{'=' * 72}")
        print(f"  {'題':<4}{'最佳解':>8}{'最佳解個數':>14}{'空間':>16}{'命中率':>12}")
        for p, best, hits, seen, rate in summary:
            print(f"  {p:<4}{best:>8}{hits:>14,}{seen:>16,}{rate:>11.6%}")


if __name__ == "__main__":
    main()
