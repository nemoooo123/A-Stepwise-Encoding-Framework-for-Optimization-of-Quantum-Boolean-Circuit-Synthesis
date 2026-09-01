"""Builds the final deliverables once (or while) the batch run has results:
  - summary_stats.xlsx  (bit, problem, algorithm, mean, std, best)
  - per-(problem, algorithm) convergence PNGs
  - per-problem combined (all algorithms) convergence PNGs
  - dashboard_data.json  (compact data for the HTML report)

Safe to run at any time; only processes combos that have finished
(fitness_history.npy + summary.json present).
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_ROOT = "batch_results"
PLOTS_DIR = os.path.join(OUT_ROOT, "plots")

ALGO_ORDER = ["AE-QTS", "QTS", "QEA", "GA", "DE", "TS", "PSO", "WOA", "ABC"]
ALGO_COLORS = {
    "AE-QTS": "#4C72B0", "QTS": "#DD8452", "QEA": "#55A868", "GA": "#C44E52",
    "DE": "#8172B2", "TS": "#937860", "PSO": "#DA8BC3", "WOA": "#8C8C8C", "ABC": "#CCB974",
}


def find_finished_combos():
    combos = []
    for summary_path in glob.glob(os.path.join(OUT_ROOT, "*bit", "*", "*", "summary.json")):
        combo_dir = os.path.dirname(summary_path)
        npy_path = os.path.join(combo_dir, "fitness_history.npy")
        if os.path.exists(npy_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            combos.append((summary, npy_path))
    return combos


def build_stats_excel(combos, out_path):
    rows = []
    for summary, _ in combos:
        rows.append({
            "Bit": summary["bit"],
            "Problem": summary["problem"],
            "Algorithm": summary["algo"],
            "Mean": round(summary["mean_best"], 4),
            "Std": round(summary["std_best"], 4),
            "Best (global min gate count)": summary["global_best"],
            "Avg Time / Experiment (s)": round(summary["avg_time_per_experiment"], 4),
            "Best Circuit": summary["best_circuit"],
        })
    df = pd.DataFrame(rows)
    algo_rank = {a: i for i, a in enumerate(ALGO_ORDER)}
    df["_algo_rank"] = df["Algorithm"].map(algo_rank)
    df = df.sort_values(["Bit", "Problem", "_algo_rank"]).drop(columns="_algo_rank")
    df.to_excel(out_path, index=False)
    return df


def build_plots_and_dashboard_data(combos):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    by_problem = {}
    for summary, npy_path in combos:
        key = (summary["bit"], summary["problem"])
        by_problem.setdefault(key, {})[summary["algo"]] = (summary, npy_path)

    dashboard = {"problems": []}

    for (bit, problem), algos in sorted(by_problem.items()):
        problem_dir = os.path.join(PLOTS_DIR, f"{bit}bit_{problem}")
        os.makedirs(problem_dir, exist_ok=True)

        combined_fig, combined_ax = plt.subplots(figsize=(9, 6))
        problem_entry = {"bit": bit, "problem": problem, "algorithms": {}}

        for algo in ALGO_ORDER:
            if algo not in algos:
                continue
            summary, npy_path = algos[algo]
            matrix = np.load(npy_path)
            mean_curve = matrix.mean(axis=0)
            std_curve = matrix.std(axis=0)

            fig, ax = plt.subplots(figsize=(8, 5))
            gens = np.arange(1, len(mean_curve) + 1)
            ax.plot(gens, mean_curve, color=ALGO_COLORS.get(algo, "#333333"), linewidth=1.5)
            ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                             color=ALGO_COLORS.get(algo, "#333333"), alpha=0.15)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Gate Count (mean of 100 trials)")
            ax.set_title(f"{problem} ({bit}-bit) — {algo} Convergence")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(problem_dir, f"{algo}_convergence.png"), dpi=130)
            plt.close(fig)

            combined_ax.plot(gens, mean_curve, label=algo, color=ALGO_COLORS.get(algo, "#333333"), linewidth=1.3)

            # Downsample for the HTML dashboard (keep the shape, cut payload size)
            step = max(1, len(mean_curve) // 300)
            problem_entry["algorithms"][algo] = {
                "mean_curve": [round(float(v), 3) for v in mean_curve[::step]],
                "gens": [int(g) for g in gens[::step]],
                "mean_best": summary["mean_best"],
                "std_best": summary["std_best"],
                "global_best": summary["global_best"],
                "avg_time": summary["avg_time_per_experiment"],
            }

        combined_ax.set_xlabel("Generation")
        combined_ax.set_ylabel("Gate Count (mean of 100 trials)")
        combined_ax.set_title(f"{problem} ({bit}-bit) — All Algorithms")
        combined_ax.legend(fontsize=8)
        combined_ax.grid(alpha=0.3)
        combined_fig.tight_layout()
        combined_fig.savefig(os.path.join(problem_dir, "ALL_algorithms_convergence.png"), dpi=130)
        plt.close(combined_fig)

        dashboard["problems"].append(problem_entry)

    return dashboard


def main():
    combos = find_finished_combos()
    print(f"Found {len(combos)} finished combos.")
    if not combos:
        print("Nothing to process yet.")
        return

    stats_path = os.path.join(OUT_ROOT, "summary_stats.xlsx")
    df = build_stats_excel(combos, stats_path)
    print(f"Wrote {stats_path} ({len(df)} rows)")

    dashboard = build_plots_and_dashboard_data(combos)
    dashboard_path = os.path.join(OUT_ROOT, "dashboard_data.json")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f)
    print(f"Wrote {dashboard_path}")
    print(f"Plots written under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
