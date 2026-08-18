"""
在 3-bit 的所有題目上跑 AE-QTS，每題 100 次取最好，並與窮舉的 ground truth 對照。

和 exhaustive_all.py 同一套架構：
  * 預設只跑 984 個等價類代表元（超立方體對稱性），跑完展開回 40,320 題
  * multiprocessing 動態配工，不需要 bash
  * 每個 task 一個 csv，可中斷續跑

用法
----
  python aeqts_all.py --run-all --jobs 20              # 984 個代表元，每題 100 次
  python aeqts_all.py --run-all --jobs 20 --all-perms  # 不用對稱性，硬跑 40,320 題
  python aeqts_all.py --run-all --jobs 20 --limit-tasks 20   # 試跑
  python aeqts_all.py --summary AEQTS_3bit_all/all_40320.csv \
      --exhaustive Exhaustive_3bit_all/all_40320.csv   # 只重算統計與對照

輸出（AEQTS_3bit_all/）
  parts/*.csv       每個 task 的結果（含 100 次的每一次最終解）
  merged.csv        每個 task 一列
  all_40320.csv     展開回全部 40,320 題
  distribution.csv  best-of-100 的分布；有 --exhaustive 時另含成功率統計
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np

from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode
from core.AE_QTS import AE_QTS_run_single_experiment
from exhaustive_all import all_tasks

NUM_BITS = 3
DEFAULT_RUNS = 100
DEFAULT_GENS = 1000
DEFAULT_NEIGHBORS = 10
DELTA_THETA = 0.01


def _blank(rows, cols):
    return np.full((rows, cols), float("inf"))


def run_aeqts(target, runs, gens, neighbors):
    """跑 runs 次 AE-QTS，回傳每次的最終解（best-so-far at last generation）。"""
    cycles, need_gate = find_cycles(target, check_zero_gate=True)
    if not need_gate:
        return np.zeros(runs)                    # 恆等映射，0 個閘

    fitness = _blank(runs, gens)
    unique = _blank(runs, gens)
    a1 = _blank(runs, gens)
    a2 = _blank(runs, gens)
    nan = lambda: np.full((runs, gens), np.nan)
    e1, e2, e3, e4, mode = nan(), nan(), nan(), nan(), nan()

    for r in range(runs):
        # build_encode 會就地反轉/延長 cycles，所以每次都要重新 find_cycles
        cycles, _ = find_cycles(target, check_zero_gate=True)
        q1, q2, q3, q4, table, base = build_encode(cycles)
        AE_QTS_run_single_experiment(
            max_iterations=gens, rotation_cycles=cycles, num_neighbors=neighbors,
            num_bits=NUM_BITS, base_trajectory=base, experiment_id=r,
            encoding_table=table,
            qindividuals1=q1, qindividuals2=q2, qindividuals3=q3, qindividuals4=q4,
            fitness_history_matrix=fitness, unique_history_matrix=unique,
            a1_history_matrix=a1, a2_history_matrix=a2,
            entropy1_history_matrix=e1, entropy2_history_matrix=e2,
            entropy3_history_matrix=e3, entropy4_history_matrix=e4,
            mode_count_history_matrix=mode,
            target_output=target, delta_theta=DELTA_THETA)

    return fitness[:, -1]


def _worker(job):
    task, rep, orbit, runs, gens, neighbors, outdir = job
    path = os.path.join(outdir, f"t_{task}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return task, 0, True

    t0 = time.time()
    finals = run_aeqts(list(rep), runs, gens, neighbors)
    elapsed = time.time() - t0

    best = float(finals.min())
    row = [task, " ".join(map(str, rep)), orbit, runs,
           int(best), f"{finals.mean():.4f}", f"{finals.std(ddof=0):.4f}",
           int((finals == best).sum()), int(finals.max()),
           f"{elapsed:.1f}",
           " ".join(str(int(v)) for v in finals)]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(",".join(map(str, row)) + "\n")
    os.replace(tmp, path)
    return task, runs, False


def cmd_run_all(tasks, jobs, runs, gens, neighbors, outroot):
    import multiprocessing as mp

    parts = os.path.join(outroot, "parts")
    os.makedirs(parts, exist_ok=True)

    joblist = [(i, rep, len(orb), runs, gens, neighbors, parts)
               for i, (rep, orb) in enumerate(tasks)]
    done = sum(1 for j in joblist
               if os.path.exists(os.path.join(parts, f"t_{j[0]}.csv")))
    print(f"[1/4] {len(joblist):,} 個 task，每題 {runs} 次 x {gens} 代"
          f"（已完成 {done:,}）", flush=True)

    print(f"[2/4] 開跑，{jobs} 個並行", flush=True)
    t0 = time.time()
    finished = skipped = 0
    try:
        with mp.Pool(jobs) as pool:
            for task, n, was_skip in pool.imap_unordered(_worker, joblist,
                                                         chunksize=1):
                finished += 1
                skipped += 1 if was_skip else 0
                if finished % 10 == 0 or finished == len(joblist):
                    el = time.time() - t0
                    pct = finished / len(joblist)
                    eta = (el / pct - el) / 60 if pct > 0 else 0
                    print(f"      {finished:,}/{len(joblist):,}（{pct:6.1%}）"
                          f"  已跑 {el / 60:.1f} 分  剩約 {eta:.0f} 分", flush=True)
    except KeyboardInterrupt:
        print("\n      [中斷] 已完成的 task 都存好了，重跑同一行即可續跑")
        return

    print(f"      耗時 {(time.time() - t0) / 60:.1f} 分"
          f"（跳過 {skipped:,} 個已完成的）", flush=True)

    merged = os.path.join(outroot, "merged.csv")
    allcsv = os.path.join(outroot, "all_40320.csv")
    print("[3/4] 合併並展開回 40,320 題")
    cmd_merge(parts, merged)
    cmd_expand(tasks, merged, allcsv)
    print("[4/4] 統計")
    cmd_summary(allcsv, os.path.join(outroot, "distribution.csv"), None)
    print(f"\n結果在 {outroot}/")


def cmd_merge(parts_dir, out):
    rows = []
    for name in sorted(os.listdir(parts_dir)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(parts_dir, name), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(line.rstrip("\n").split(","))
    rows.sort(key=lambda r: int(r[0]))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "perm", "orbit_size", "runs", "best", "mean", "std",
                    "best_count", "worst", "seconds", "finals"])
        w.writerows(rows)
    print(f"      合併 {len(rows)} 個 task -> {out}")


def cmd_expand(tasks, merged, out):
    by_task = {}
    with open(merged, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_task[int(r["task"])] = r

    missing = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["perm", "runs", "best", "mean", "std", "best_count",
                    "worst", "rep_perm", "task", "orbit_size"])
        for i, (rep, orb) in enumerate(tasks):
            r = by_task.get(i)
            if r is None:
                missing += 1
                continue
            for member in orb:
                w.writerow([" ".join(map(str, member)), r["runs"],
                            r["best"], r["mean"], r["std"],
                            r["best_count"], r["worst"],
                            " ".join(map(str, rep)), i, len(orb)])
    print(f"      展開 -> {out}" + (f"（缺 {missing} 個 task）" if missing else ""))


def cmd_summary(path, out, exhaustive):
    """best-of-100 的分布；給了 --exhaustive 就一併算「找到最佳解」的成功率。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = len(rows)
    best_dist = Counter(int(r["best"]) for r in rows)
    runs = int(rows[0].get("runs") or 0)

    print(f"\n{'=' * 66}")
    print(f"AE-QTS best-of-{runs} 分布（{n:,} 題）")
    print(f"{'=' * 66}")
    print(f"  {'閘數':>6}{'題數':>10}{'佔比':>10}{'累積':>10}")
    cum = 0
    for g in sorted(best_dist):
        cum += best_dist[g]
        print(f"  {g:>6}{best_dist[g]:>10,}{best_dist[g] / n:>9.3%}{cum / n:>9.3%}")
    total = sum(g * c for g, c in best_dist.items())
    print(f"\n  best-of-{runs} 總和 = {total:,}    平均 = {total / n:.4f} 閘")

    opt_rows = None
    if exhaustive:
        opt = {}
        with open(exhaustive, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                opt[r["perm"]] = int(r["best_gates"])
        matched = [r for r in rows if r["perm"] in opt]
        hit = sum(1 for r in matched if int(r["best"]) == opt[r["perm"]])
        gap = Counter(int(r["best"]) - opt[r["perm"]] for r in matched)
        rate = [int(r["best_count"]) / runs for r in matched
                if int(r["best"]) == opt[r["perm"]]]

        print(f"\n{'=' * 66}")
        print(f"與窮舉 ground truth 對照（{len(matched):,} 題可對照）")
        print(f"{'=' * 66}")
        print(f"  best-of-{runs} == 窮舉最佳解: {hit:,}/{len(matched):,}"
              f"（{hit / max(len(matched), 1):.3%}）")
        print(f"  與最佳解的差距分布:")
        for d in sorted(gap):
            print(f"    +{d:<3} 閘: {gap[d]:>7,} 題（{gap[d] / len(matched):7.3%}）")
        if rate:
            arr = np.array(rate)
            print(f"\n  在找到最佳解的題目裡，{runs} 次中命中最佳解的比例:")
            print(f"    平均 {arr.mean():.1%}  中位數 {np.median(arr):.1%}"
                  f"  最低 {arr.min():.1%}  最高 {arr.max():.1%}")
            for lo, hi in ((0, .05), (.05, .25), (.25, .5), (.5, .75), (.75, 1.01)):
                c = int(((arr >= lo) & (arr < hi)).sum())
                print(f"    {lo:.0%}-{hi:.0%}: {c:,} 題")
        opt_rows = (hit, len(matched), gap)

    if n != 40320:
        print(f"\n  [注意] 只有 {n:,} 題，不是 40,320 —— 有 task 沒跑完")

    if out:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["best_gates", "num_problems", "share"])
            for g in sorted(best_dist):
                w.writerow([g, best_dist[g], f"{best_dist[g] / n:.6f}"])
            w.writerow([])
            w.writerow(["total", total])
            w.writerow(["mean", f"{total / n:.6f}"])
            w.writerow(["num_problems", n])
            if opt_rows:
                hit, m, gap = opt_rows
                w.writerow([])
                w.writerow(["matched_vs_exhaustive", m])
                w.writerow(["found_optimum", hit])
                w.writerow(["found_optimum_share", f"{hit / max(m, 1):.6f}"])
                for d in sorted(gap):
                    w.writerow([f"gap_plus_{d}", gap[d]])
        print(f"\n  分布表 -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--gens", type=int, default=DEFAULT_GENS)
    ap.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    ap.add_argument("--outdir", default="AEQTS_3bit_all")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--all-perms", action="store_true",
                    help="不使用對稱性，逐一跑全部 40,320 個排列（約 41 倍時間）")
    ap.add_argument("--summary")
    ap.add_argument("--exhaustive", help="窮舉結果 all_40320.csv，用來算成功率")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.summary:
        cmd_summary(args.summary, args.out, args.exhaustive)
        return

    if args.all_perms:
        from itertools import permutations
        tasks = [(p, [p]) for p in permutations(range(8))]
        print(f"[模式] 不用對稱性，全部 {len(tasks):,} 個排列", file=sys.stderr)
    else:
        tasks = all_tasks()

    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
        print(f"[試跑] 只用前 {len(tasks)} 個 task", file=sys.stderr)

    if args.run_all:
        cmd_run_all(tasks, args.jobs, args.runs, args.gens,
                    args.neighbors, args.outdir)
    else:
        ap.error("要指定 --run-all 或 --summary")


if __name__ == "__main__":
    main()
