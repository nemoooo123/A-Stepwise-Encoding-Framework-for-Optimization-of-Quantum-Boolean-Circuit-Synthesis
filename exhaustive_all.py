"""
窮舉 3-bit 的「所有題目」（8! = 40,320 個排列）的編碼空間最佳解。

三個讓它從 663 核心小時降到約 29 核心小時的作法：

1. 超立方體對稱性（48 倍 -> 實際 22.9 倍）
   把 target 用 A(x) = 對 bit 位置排列後 XOR c 共軛（3! * 8 = 48 個自同構），
   所有 hamming 距離都不變，所以編碼空間同構、最佳閘數相同。
   40,320 個排列因此收斂成 984 個等價類，只需跑代表元，再展開回全部。
   （已用 exhaustive_search 對 48 個共軛逐一驗證過最佳值完全一致）

2. 直接呼叫 synthesize_route
   跳過 decode_and_synthesize 每個解都要做的 bin()/zfill()/int(x,2)，
   canonical 模式本來就知道解碼後的整數，不必繞一圈。

3. 可索引的窮舉（--part/--nparts）
   空間被拆成「(L1 順序, L2 entry 組合)」的區塊，每個區塊是各 slot 的直積，
   所以可以用全域索引直接切片。最大的那題有 4.78 億個解、單核要 8.75 小時，
   靠這個才能把它也分散到多顆 CPU 上。

用法
----
  python exhaustive_all.py --list-tasks            # 列出 984 個代表元與空間大小
  python exhaustive_all.py --plan --parts-target N # 產生給 xargs 的工作清單
  python exhaustive_all.py --task 7                # 跑單一代表元
  python exhaustive_all.py --task 7 --part 0 --nparts 8
  python exhaustive_all.py --expand results.csv    # 把代表元結果展開回 40,320 題
一般不用手動跑，用 run_exhaustive_3bit.sh。
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from itertools import combinations, islice, permutations, product

from utils.topology import synthesize_route
from exhaustive_search import describe, space_size, used_steps

N_STATES, NUM_BITS = 8, 3


# ---------- 超立方體自同構與等價類 ----------

def build_autos():
    """回傳 48 個自同構的查表 (list[8])。"""
    maps = []
    for sigma in permutations(range(NUM_BITS)):
        for c in range(N_STATES):
            m = []
            for x in range(N_STATES):
                bits = [(x >> (NUM_BITS - 1 - i)) & 1 for i in range(NUM_BITS)]
                v = 0
                for i in range(NUM_BITS):
                    v = (v << 1) | bits[sigma[i]]
                m.append(v ^ c)
            maps.append(m)
    return maps


AUTOS = build_autos()
AUTO_INV = [[m.index(y) for y in range(N_STATES)] for m in AUTOS]


def conjugate(target, k):
    """target' = A o target o A^-1，A = AUTOS[k]。"""
    m, inv = AUTOS[k], AUTO_INV[k]
    return tuple(m[target[inv[y]]] for y in range(N_STATES))


def orbit_of(target):
    return {conjugate(target, k) for k in range(len(AUTOS))}


def all_tasks():
    """回傳 [(rep_perm, orbit_members)]，代表元依字典序取每個等價類最小者。"""
    seen, tasks = set(), []
    for perm in permutations(range(N_STATES)):
        if perm in seen:
            continue
        orb = orbit_of(perm)
        seen |= orb
        tasks.append((min(orb), sorted(orb)))
    return tasks


# ---------- 可索引的窮舉計畫 ----------

def make_l1_weights(order, n):
    """order = cycle 處理順序（優先度高在前）-> 解碼後的權重值。"""
    w = [0] * n
    for k, c in enumerate(order):
        w[c] = n - 1 - k
    return w


def make_l3_weights(perm, d):
    """perm = bit-flip 先後順序 -> 解碼後的權重值。"""
    w = [0] * d
    for k, sub in enumerate(perm):
        w[sub] = d - 1 - k
    return w


def balanced_vectors(length):
    half = length // 2
    out = []
    for ones in combinations(range(length), half):
        v = [0] * length
        for i in ones:
            v[i] = 1
        out.append(v)
    return out


def build_plan(info):
    """
    把整個 canonical 空間切成一串區塊，每個區塊是 slots 的直積。
    回傳 (blocks, total)；blocks 內每項為 dict，start 是該區塊的全域起始索引。
    """
    n = info["n_cycles"]
    cycles = info["cycles"]
    l1_list = [make_l1_weights(o, n) for o in permutations(range(n))]
    entry_ranges = [range(info["limits"][c]) for c in range(n)]

    blocks, total = [], 0
    for l1 in l1_list:
        for entries in product(*entry_ranges):
            slots, layout = [], []
            for c in range(n):
                for s in used_steps(entries[c], cycles[c]):
                    st = cycles[c]["steps"][s]
                    if st["d"] == 1:
                        continue
                    slots.append([make_l3_weights(p, st["d"])
                                  for p in permutations(range(st["d"]))])
                    slots.append(balanced_vectors(st["b4"]))
                    layout.append((c, s))
            size = 1
            for sl in slots:
                size *= len(sl)
            blocks.append({"l1": l1, "entries": list(entries), "slots": slots,
                           "layout": layout, "size": size, "start": total})
            total += size
    return blocks, total


def evaluate_range(info, blocks, lo, hi, progress_every=0):
    """評估全域索引 [lo, hi) 的解，回傳 (最小閘數, 該值出現次數, 已評估數)。"""
    n = info["n_cycles"]
    cycles = info["cycles"]
    traj = info["traj"]
    best, best_count, seen = None, 0, 0
    t0 = time.time()

    # 每個 cycle 的 step 數，用來建 l3/l4 骨架
    n_steps = [len(cycles[c]["steps"]) for c in range(n)]

    for blk in blocks:
        b_lo = blk["start"]
        b_hi = b_lo + blk["size"]
        if b_hi <= lo:
            continue
        if b_lo >= hi:
            break
        off_lo = max(lo - b_lo, 0)
        off_hi = min(hi - b_lo, blk["size"])

        l1, entries, layout = blk["l1"], blk["entries"], blk["layout"]
        for combo in islice(product(*blk["slots"]), off_lo, off_hi):
            l3 = [[[999] for _ in range(n_steps[c])] for c in range(n)]
            l4 = [[[0] for _ in range(n_steps[c])] for c in range(n)]
            for k, (c, s) in enumerate(layout):
                l3[c][s] = combo[2 * k]
                l4[c][s] = combo[2 * k + 1]
            circuit = synthesize_route(l1, entries, l3, l4, NUM_BITS, traj, 1)[0]
            gc = len(circuit)
            seen += 1
            if best is None or gc < best:
                best, best_count = gc, 1
            elif gc == best:
                best_count += 1
            if progress_every and seen % progress_every == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                print(f"    {seen:,}/{hi - lo:,}  min={best}  {rate:,.0f}/s",
                      file=sys.stderr, flush=True)
    return best, best_count, seen


# ---------- 子命令 ----------

def cmd_list_tasks(tasks):
    w = csv.writer(sys.stdout)
    w.writerow(["task", "perm", "orbit_size", "n_cycles", "space"])
    for i, (rep, orb) in enumerate(tasks):
        info = describe(list(rep))
        sp = 1 if info["n_cycles"] == 0 else space_size(info, "canonical")
        w.writerow([i, " ".join(map(str, rep)), len(orb), info["n_cycles"], sp])


def cmd_plan(tasks, parts_target):
    """
    產生工作清單（每行 "task part nparts"），大題目會被切成多份。
    parts_target 是希望的每份工作量（解的個數）。
    """
    rows = []
    for i, (rep, _) in enumerate(tasks):
        info = describe(list(rep))
        sp = 1 if info["n_cycles"] == 0 else space_size(info, "canonical")
        nparts = max(1, -(-sp // parts_target))
        for p in range(nparts):
            rows.append((sp // nparts, i, p, nparts))
    rows.sort(reverse=True)          # 大的先跑，xargs 動態配工時尾巴才不會拖
    for _, i, p, nparts in rows:
        print(f"{i} {p} {nparts}")


def cmd_task(tasks, task, part, nparts, out):
    rep, orb = tasks[task]
    info = describe(list(rep))
    if info["n_cycles"] == 0:
        best, count, seen = 0, 1, 1
    else:
        blocks, total = build_plan(info)
        lo = total * part // nparts
        hi = total * (part + 1) // nparts
        best, count, seen = evaluate_range(info, blocks, lo, hi)

    row = [task, " ".join(map(str, rep)), len(orb), info["n_cycles"],
           part, nparts, "" if best is None else best, count, seen]
    line = ",".join(map(str, row))
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    else:
        print(line)


def cmd_merge(paths, out):
    """把所有 part 的結果合併成每個 task 一列。paths 可以是檔案或目錄。"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if f.endswith(".csv"))
        else:
            files.append(p)

    agg = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                f = line.split(",")
                t = int(f[0])
                best = None if f[6] == "" else int(f[6])
                cnt, seen = int(f[7]), int(f[8])
                cur = agg.setdefault(t, {"perm": f[1], "orbit": int(f[2]),
                                         "ncyc": int(f[3]), "best": best,
                                         "count": 0, "seen": 0, "parts": 0})
                cur["seen"] += seen
                cur["parts"] += 1
                if best is not None:
                    if cur["best"] is None or best < cur["best"]:
                        cur["best"], cur["count"] = best, cnt
                    elif best == cur["best"]:
                        cur["count"] += cnt

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "perm", "orbit_size", "n_cycles",
                    "best_gates", "best_count", "space", "parts"])
        for t in sorted(agg):
            a = agg[t]
            w.writerow([t, a["perm"], a["orbit"], a["ncyc"],
                        a["best"], a["count"], a["seen"], a["parts"]])
    print(f"合併 {len(agg)} 個 task -> {out}")


def cmd_expand(tasks, merged, out):
    """把 984 個代表元的結果展開成 40,320 題。"""
    by_task = {}
    with open(merged, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_task[int(r["task"])] = r

    missing = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["perm", "best_gates", "rep_perm", "task", "orbit_size"])
        for i, (rep, orb) in enumerate(tasks):
            r = by_task.get(i)
            if r is None:
                missing += 1
                continue
            for member in orb:
                w.writerow([" ".join(map(str, member)), r["best_gates"],
                            " ".join(map(str, rep)), i, len(orb)])
    print(f"展開完成 -> {out}" + (f"（缺 {missing} 個 task）" if missing else ""))


def cmd_summary(path, out):
    """統計 40,320 題的窮舉最佳解分佈與總和。"""
    gates, rows = Counter(), 0
    per_perm = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = int(r["best_gates"])
            gates[g] += 1
            per_perm[r["perm"]] = g
            rows += 1

    total = sum(g * c for g, c in gates.items())
    print(f"\n{'=' * 60}")
    print(f"3-bit 全部 {rows:,} 題的窮舉最佳解分佈")
    print(f"{'=' * 60}")
    print(f"  {'閘數':>6}{'題數':>10}{'佔比':>10}{'累積':>10}")
    cum = 0
    for g in sorted(gates):
        cum += gates[g]
        print(f"  {g:>6}{gates[g]:>10,}{gates[g] / rows:>9.3%}{cum / rows:>9.3%}")
    print(f"\n  最佳解總和 = {total:,}")
    print(f"  平均       = {total / rows:.4f} 閘")
    print(f"  最小 / 最大 = {min(gates)} / {max(gates)}")
    print(f"  相異閘數值  = {sorted(gates)}")

    if rows != 40320:
        print(f"\n  [注意] 只有 {rows:,} 題，不是 40,320 —— 有分片沒跑完")

    # 對照 data_loader 內建的 5 題
    try:
        from utils.data_loader import DataLoader
        loader = DataLoader()
        print("\n  對照 data_loader 內建的 5 題：")
        for i, tgt in enumerate(loader.data_map[3], start=1):
            key = " ".join(map(str, tgt))
            print(f"    第 {i} 題 {key} -> {per_perm.get(key, '?')} 閘")
    except Exception as exc:                                  # pragma: no cover
        print(f"  [跳過內建題目對照] {exc}")

    if out:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["best_gates", "num_problems", "share"])
            for g in sorted(gates):
                w.writerow([g, gates[g], f"{gates[g] / rows:.6f}"])
            w.writerow([])
            w.writerow(["total_gates_sum", total])
            w.writerow(["mean_gates", f"{total / rows:.6f}"])
            w.writerow(["num_problems", rows])
        print(f"\n  分佈表 -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--parts-target", type=int, default=20_000_000)
    ap.add_argument("--task", type=int)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--expand")
    ap.add_argument("--summary")
    args = ap.parse_args()

    tasks = all_tasks()

    if args.list_tasks:
        cmd_list_tasks(tasks)
    elif args.plan:
        cmd_plan(tasks, args.parts_target)
    elif args.merge:
        cmd_merge(args.merge, args.out or "merged.csv")
    elif args.expand:
        cmd_expand(tasks, args.expand, args.out or "all_40320.csv")
    elif args.summary:
        cmd_summary(args.summary, args.out)
    elif args.task is not None:
        cmd_task(tasks, args.task, args.part, args.nparts, args.out)
    else:
        ap.error("要指定 --list-tasks / --plan / --task / --merge / --expand / --summary")


if __name__ == "__main__":
    main()
