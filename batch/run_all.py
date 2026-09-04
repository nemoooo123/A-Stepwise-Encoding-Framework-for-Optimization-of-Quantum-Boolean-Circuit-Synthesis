"""Orchestrator: runs every (problem, algorithm) combo across all CPU cores.

Usage:
    python -m batch.run_all
"""
import os
import sys
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch.runner import run_combo, ALGOS

OUT_ROOT = "batch_results"

PROBLEMS = [
    (4, 1, "hwb4"),
    (4, 2, "4_49"),
    (4, 3, "4b15g_1"),
    (4, 4, "4b15g_2"),
    (4, 5, "4b15g_3"),
    (4, 6, "4b15g_4"),
    (4, 7, "4b15g_5"),
    (4, 8, "nth_prime4_inc"),
    (5, 6, "nth_prime5_inc"),
    (5, 7, "hwb5"),
    (6, 6, "permanent2x2"),
    (6, 7, "nth_prime6_inc"),
    (6, 8, "hwb6"),
]

NUM_EXPERIMENTS = 100
MAX_ITERATIONS = 3000
NUM_NEIGHBORS = 10


def _worker(task):
    bit, problem_idx, problem_name, algo_choice = task
    return run_combo(
        bit=bit,
        problem_idx=problem_idx,
        problem_name=problem_name,
        algo_choice=algo_choice,
        num_experiments=NUM_EXPERIMENTS,
        max_iterations=MAX_ITERATIONS,
        num_neighbors=NUM_NEIGHBORS,
        out_root=OUT_ROOT,
    )


def main():
    tasks = []
    for bit, problem_idx, problem_name in PROBLEMS:
        for algo_choice in ALGOS:
            tasks.append((bit, problem_idx, problem_name, algo_choice))

    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(OUT_ROOT, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "problems": PROBLEMS,
            "algos": ALGOS,
            "num_experiments": NUM_EXPERIMENTS,
            "max_iterations": MAX_ITERATIONS,
            "total_tasks": len(tasks),
            "started_at": time.time(),
        }, f, indent=2)

    n_workers = mp.cpu_count()
    print(f"Launching {len(tasks)} tasks across {n_workers} worker processes...", flush=True)

    results = []
    summary_path = os.path.join(OUT_ROOT, "all_summaries.json")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker, task): task for task in tasks}
        done_count = 0
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                summary = fut.result()
                results.append(summary)
            except Exception as e:
                results.append({
                    "bit": task[0], "problem": task[2], "algo": ALGOS[task[3]],
                    "error": str(e),
                })
            done_count += 1
            elapsed = time.time() - t0
            print(f"[{done_count}/{len(tasks)}] finished {task[2]} / {ALGOS[task[3]]}  (elapsed {elapsed:.1f}s)", flush=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    total_elapsed = time.time() - t0
    print(f"All {len(tasks)} tasks complete in {total_elapsed:.1f}s", flush=True)
    with open(manifest_path, "r+", encoding="utf-8") as f:
        manifest = json.load(f)
        manifest["finished_at"] = time.time()
        manifest["total_elapsed"] = total_elapsed
        f.seek(0)
        json.dump(manifest, f, indent=2)
        f.truncate()


if __name__ == "__main__":
    main()
