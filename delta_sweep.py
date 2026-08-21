"""
AE-QTS delta_theta sweep for 6-bit problem #1.

Runs NUM_TRIALS independent trials for each delta_theta value in DELTAS,
in parallel across MAX_WORKERS processes, then aggregates and plots the
averaged convergence curves.

Usage:
    python delta_sweep.py              # run the full sweep, then plot
    python delta_sweep.py --plot-only  # re-plot from already-saved data
"""
import os
import sys
import time
import random
import argparse

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

NUM_BITS = 6
PROBLEM_IDX = 1
MAX_ITERATIONS = 1000
NUM_NEIGHBORS = 10
NUM_TRIALS = 100
DELTAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
MAX_WORKERS = 14  # 16 logical processors on this machine, 2 left free

OUT_DIR = os.path.join(PROJECT_DIR, "exp", f"{NUM_BITS}_bit", "AE-QTS_delta_sweep")

# Sequential "blue" ramp from references/palette.md (dataviz skill), ordinal steps
# 250..700, one step per delta value from smallest (lightest) to largest (darkest).
SEQUENTIAL_BLUE = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Categorical palette (8 hues, fixed order, CVD-safe adjacent pairs) from
# references/palette.md. Only 8 slots validate safely, so the 9th/10th delta
# value reuse slot 1/2's hue with a dashed linestyle (composite encoding)
# instead of inventing new hues.
CATEGORICAL_8 = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
LINE_COLORS = CATEGORICAL_8 + CATEGORICAL_8[:2]
LINE_STYLES = ["-"] * 8 + ["--"] * 2


def run_one_trial(delta_theta, trial_id):
    from utils.data_loader import DataLoader
    from utils.init_state import find_cycles, build_encode
    from core.AE_QTS import AE_QTS_run_single_experiment

    seed = hash((NUM_BITS, PROBLEM_IDX, delta_theta, trial_id)) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)

    loader = DataLoader()
    target_output = loader.get_output(NUM_BITS, PROBLEM_IDX)
    cycles, is_gate_required = find_cycles(target_output, check_zero_gate=True)

    if not is_gate_required:
        return delta_theta, trial_id, np.zeros(MAX_ITERATIONS), 0.0, 0.0

    q1, q2, q3, q4, enc, traj = build_encode(cycles)
    fh = np.full((1, MAX_ITERATIONS), float("inf"))
    uh = np.full((1, MAX_ITERATIONS), float("inf"))
    a1 = np.full((1, MAX_ITERATIONS), float("inf"))
    a2 = np.full((1, MAX_ITERATIONS), float("inf"))
    e1 = np.full((1, MAX_ITERATIONS), np.nan)
    e2 = np.full((1, MAX_ITERATIONS), np.nan)
    e3 = np.full((1, MAX_ITERATIONS), np.nan)
    e4 = np.full((1, MAX_ITERATIONS), np.nan)
    mc = np.full((1, MAX_ITERATIONS), np.nan)

    t0 = time.time()
    result = AE_QTS_run_single_experiment(
        max_iterations=MAX_ITERATIONS, rotation_cycles=cycles, num_neighbors=NUM_NEIGHBORS,
        num_bits=NUM_BITS, base_trajectory=traj, experiment_id=0, encoding_table=enc,
        qindividuals1=q1, qindividuals2=q2, qindividuals3=q3, qindividuals4=q4,
        fitness_history_matrix=fh, unique_history_matrix=uh, a1_history_matrix=a1, a2_history_matrix=a2,
        entropy1_history_matrix=e1, entropy2_history_matrix=e2, entropy3_history_matrix=e3, entropy4_history_matrix=e4,
        mode_count_history_matrix=mc, target_output=target_output, delta_theta=delta_theta,
    )
    elapsed = time.time() - t0
    final_best_gate = result[9]
    return delta_theta, trial_id, fh[0].copy(), elapsed, float(final_best_gate)


def run_sweep():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(d, t) for d in DELTAS for t in range(NUM_TRIALS)]
    total = len(jobs)
    print(f"Total jobs: {total} ({len(DELTAS)} deltas x {NUM_TRIALS} trials), workers={MAX_WORKERS}")

    curves = {d: np.full((NUM_TRIALS, MAX_ITERATIONS), np.nan) for d in DELTAS}
    finals = {d: [None] * NUM_TRIALS for d in DELTAS}
    times = {d: [None] * NUM_TRIALS for d in DELTAS}

    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one_trial, d, t): (d, t) for d, t in jobs}
        for fut in as_completed(futs):
            delta_theta, trial_id, curve, elapsed, final_gate = fut.result()
            curves[delta_theta][trial_id] = curve
            finals[delta_theta][trial_id] = final_gate
            times[delta_theta][trial_id] = elapsed
            done += 1
            if done % 25 == 0 or done == total:
                elapsed_total = time.time() - t_start
                eta = elapsed_total / done * (total - done)
                print(f"[{done}/{total}] {done/total*100:.1f}% | elapsed {elapsed_total:.0f}s | ETA {eta:.0f}s")

    import pandas as pd
    summary_rows = []
    for d in DELTAS:
        np.savez_compressed(
            os.path.join(OUT_DIR, f"delta_{d}.npz"),
            fitness_history=curves[d],
            final_gates=np.array(finals[d], dtype=float),
            exec_times=np.array(times[d], dtype=float),
        )
        mean_curve = np.mean(curves[d], axis=0)
        std_curve = np.std(curves[d], axis=0)
        np.savez_compressed(os.path.join(OUT_DIR, f"delta_{d}_curve.npz"), mean=mean_curve, std=std_curve)
        summary_rows.append({
            "delta": d,
            "final_mean": mean_curve[-1],
            "final_std": std_curve[-1],
            "best_of_trials": float(np.min(finals[d])),
            "mean_exec_time_s": float(np.mean(times[d])),
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")
    print(f"Raw + summary data saved under: {OUT_DIR}")


def plot_sweep():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    iterations = np.arange(1, MAX_ITERATIONS + 1)
    curves_mean = {}
    curves_std = {}
    for d in DELTAS:
        data = np.load(os.path.join(OUT_DIR, f"delta_{d}_curve.npz"))
        curves_mean[d] = data["mean"]
        curves_std[d] = data["std"]

    summary = pd.read_csv(os.path.join(OUT_DIR, "summary.csv"))

    # --- Combined convergence curves (linear y) ---
    plt.figure(figsize=(9, 6))
    for color, style, d in zip(LINE_COLORS, LINE_STYLES, DELTAS):
        plt.plot(iterations, curves_mean[d], label=f"δ={d}", color=color, linestyle=style, linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("Gate Count (mean of {} trials)".format(NUM_TRIALS))
    plt.title(f"AE-QTS Convergence vs delta_theta (6-bit, problem #{PROBLEM_IDX})")
    plt.legend(title="delta_theta", fontsize=8, ncol=2)
    plt.tight_layout()
    linear_path = os.path.join(OUT_DIR, "convergence_linear.png")
    plt.savefig(linear_path, dpi=150)
    plt.close()

    # --- Combined convergence curves (log y, clearer separation of tails) ---
    plt.figure(figsize=(9, 6))
    for color, style, d in zip(LINE_COLORS, LINE_STYLES, DELTAS):
        plt.plot(iterations, curves_mean[d], label=f"δ={d}", color=color, linestyle=style, linewidth=1.8)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Gate Count (mean of {} trials, log scale)".format(NUM_TRIALS))
    plt.title(f"AE-QTS Convergence vs delta_theta (6-bit, problem #{PROBLEM_IDX}), log y")
    plt.legend(title="delta_theta", fontsize=8, ncol=2)
    plt.tight_layout()
    log_path = os.path.join(OUT_DIR, "convergence_logy.png")
    plt.savefig(log_path, dpi=150)
    plt.close()

    # --- Final performance vs delta (summary) ---
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        summary["delta"], summary["final_mean"], yerr=summary["final_std"],
        marker="o", color=SEQUENTIAL_BLUE[4], ecolor=SEQUENTIAL_BLUE[8],
        capsize=3, linewidth=1.6,
    )
    plt.xscale("log")
    plt.xlabel("delta_theta (log scale)")
    plt.ylabel(f"Final Gate Count (mean ± std, gen {MAX_ITERATIONS})")
    plt.title(f"AE-QTS Final Result vs delta_theta (6-bit, problem #{PROBLEM_IDX})")
    plt.tight_layout()
    summary_path = os.path.join(OUT_DIR, "final_vs_delta.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()

    print(f"Saved: {linear_path}")
    print(f"Saved: {log_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="skip the sweep, only re-plot saved data")
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep()
    plot_sweep()
