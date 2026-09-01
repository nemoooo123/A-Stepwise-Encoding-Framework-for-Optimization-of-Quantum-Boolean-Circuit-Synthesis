"""Prints a live snapshot of batch-run progress.

Usage:
    python -m batch.check_progress
"""
import os
import json
import time
import glob

OUT_ROOT = "batch_results"


def main():
    progress_dir = os.path.join(OUT_ROOT, "progress")
    files = sorted(glob.glob(os.path.join(progress_dir, "*.json")))
    if not files:
        print("No progress files yet — the batch run may not have started.")
        return

    rows = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                rows.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue

    total_combos = len(rows)
    done_combos = sum(1 for r in rows if r.get("status") == "done")
    total_exp = sum(r.get("total", 0) for r in rows)
    done_exp = sum(r.get("completed", 0) for r in rows)

    print(f"Combos: {done_combos}/{total_combos} done   |   Experiments: {done_exp}/{total_exp} "
          f"({100.0*done_exp/max(total_exp,1):.1f}%)")
    print("-" * 78)
    print(f"{'bit':<4}{'problem':<18}{'algo':<10}{'progress':<12}{'status':<10}{'elapsed(s)':<10}")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: (r.get("bit", 0), r.get("problem", ""), r.get("algo", ""))):
        elapsed = r.get("last_update", 0) - r.get("start_time", 0)
        print(f"{r.get('bit',''):<4}{r.get('problem',''):<18}{r.get('algo',''):<10}"
              f"{str(r.get('completed',0))+'/'+str(r.get('total',0)):<12}"
              f"{r.get('status',''):<10}{elapsed:<10.1f}")


if __name__ == "__main__":
    main()
