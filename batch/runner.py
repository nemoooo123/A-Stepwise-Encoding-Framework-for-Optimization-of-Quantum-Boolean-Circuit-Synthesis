"""Reusable single (bit, problem, algorithm) experiment-set runner.

Mirrors main.py's per-algorithm dispatch, but generalized so it can be
driven by an orchestrator across many (bit, problem, algorithm) combos in
parallel, with progress reporting and full per-generation data retained.
"""
import os
import json
import time
import numpy as np
import pandas as pd

from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode
from core.AE_QTS import AE_QTS_run_single_experiment
from core.DE import DE_run_single_experiment
from core.PSO import PSO_run_single_experiment
from core.TS import TS_run_single_experiment
from core.QTS import QTS_run_single_experiment
from core.GA import GA_run_single_experiment
from core.ABC import ABC_run_single_experiment
from core.WOA import WOA_run_single_experiment
from core.QEA import QEA_run_single_experiment

ALGOS = {
    1: "AE-QTS",
    2: "QTS",
    3: "QEA",
    4: "GA",
    5: "DE",
    6: "TS",
    7: "PSO",
    8: "WOA",
    9: "ABC",
}

# Fixed, tuned hyperparameters (kept identical to main.py)
EXTRA_KWARGS = {
    1: {"delta_theta": 0.01},
    2: {"delta_theta": 0.01},
    3: {"delta_theta": 0.0005},
    4: {"k": 5, "pc": 0.8, "pm": 0.005},
    5: {"CR": 0.01},
    6: {"tabu_size": 45},
    7: {"w": 0.6, "c1": 3.0, "c2": 1.0},
    8: {"b": 0.45},
    9: {"limit": 40},
}

RUN_FUNCS = {
    1: AE_QTS_run_single_experiment,
    2: QTS_run_single_experiment,
    3: QEA_run_single_experiment,
    4: GA_run_single_experiment,
    5: DE_run_single_experiment,
    6: TS_run_single_experiment,
    7: PSO_run_single_experiment,
    8: WOA_run_single_experiment,
    9: ABC_run_single_experiment,
}

# algos 4-9 take the four encode arrays under the name pop_matrixN instead
# of qindividualsN, but it's the same tuple returned by build_encode.
POP_STYLE_ALGOS = {4, 5, 6, 7, 8, 9}


def _write_progress(progress_path, payload):
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, progress_path)


def run_combo(
    bit,
    problem_idx,
    problem_name,
    algo_choice,
    num_experiments=100,
    max_iterations=3000,
    num_neighbors=10,
    out_root="batch_results",
    progress_root=None,
):
    algo_name = ALGOS[algo_choice]
    progress_root = progress_root or os.path.join(out_root, "progress")
    os.makedirs(progress_root, exist_ok=True)
    progress_path = os.path.join(progress_root, f"{bit}bit_{problem_name}_{algo_name}.json")

    save_dir = os.path.join(out_root, f"{bit}bit", problem_name, algo_name)
    os.makedirs(save_dir, exist_ok=True)

    loader = DataLoader()
    target_output = loader.get_output(bit, problem_idx)
    if not target_output:
        raise ValueError(f"Problem not found: bit={bit} idx={problem_idx}")

    fitness_history_matrix = np.full((num_experiments, max_iterations), float("inf"))
    best_scores_per_experiment = []
    best_circuit_per_experiment = []
    execution_times = []

    start_time = time.time()
    _write_progress(progress_path, {
        "bit": bit, "problem": problem_name, "algo": algo_name,
        "completed": 0, "total": num_experiments, "status": "running",
        "start_time": start_time, "last_update": start_time,
    })

    run_func = RUN_FUNCS[algo_choice]
    extra_kwargs = EXTRA_KWARGS[algo_choice]

    for r in range(num_experiments):
        exp_start = time.time()
        cycles, is_gate_required = find_cycles(target_output, check_zero_gate=True)

        if not is_gate_required:
            best_scores_per_experiment.append(0)
            fitness_history_matrix[r, :] = 0
            best_circuit_per_experiment.append([])
            execution_times.append(time.time() - exp_start)
        else:
            e1, e2, e3, e4, encoding_table, trajectory_base = build_encode(cycles)

            kwargs = dict(
                max_iterations=max_iterations,
                rotation_cycles=cycles,
                num_neighbors=num_neighbors,
                num_bits=bit,
                base_trajectory=trajectory_base,
                experiment_id=r,
                encoding_table=encoding_table,
                fitness_history_matrix=fitness_history_matrix,
                target_output=target_output,
                **extra_kwargs,
            )
            if algo_choice in POP_STYLE_ALGOS:
                kwargs.update(pop_matrix1=e1, pop_matrix2=e2, pop_matrix3=e3, pop_matrix4=e4)
            else:
                kwargs.update(qindividuals1=e1, qindividuals2=e2, qindividuals3=e3, qindividuals4=e4)

            fitness_history_matrix, final_best_gate, best_circuit_this_run = run_func(**kwargs)

            execution_times.append(time.time() - exp_start)
            best_circuit_per_experiment.append(best_circuit_this_run)
            best_scores_per_experiment.append(final_best_gate)

        now = time.time()
        _write_progress(progress_path, {
            "bit": bit, "problem": problem_name, "algo": algo_name,
            "completed": r + 1, "total": num_experiments, "status": "running",
            "start_time": start_time, "last_update": now,
        })

    total_elapsed = time.time() - start_time

    # --- Persist raw per-generation data (fast reload for plotting) ---
    np.save(os.path.join(save_dir, "fitness_history.npy"), fitness_history_matrix)
    with open(os.path.join(save_dir, "best_circuits.json"), "w", encoding="utf-8") as f:
        json.dump([str(c) for c in best_circuit_per_experiment], f)

    average_convergence_curve = np.mean(fitness_history_matrix, axis=0)
    std_convergence_curve = np.std(fitness_history_matrix, axis=0)
    global_min_gate = min(best_scores_per_experiment)
    best_exp_index = best_scores_per_experiment.index(global_min_gate)
    absolute_best_circuit = best_circuit_per_experiment[best_exp_index]

    mean_best = float(np.mean(best_scores_per_experiment))
    std_best = float(np.std(best_scores_per_experiment))

    summary = {
        "bit": bit,
        "problem_idx": problem_idx,
        "problem": problem_name,
        "algo": algo_name,
        "num_experiments": num_experiments,
        "max_iterations": max_iterations,
        "mean_best": mean_best,
        "std_best": std_best,
        "global_best": float(global_min_gate),
        "best_circuit": str(absolute_best_circuit),
        "avg_time_per_experiment": float(np.mean(execution_times)),
        "total_elapsed": total_elapsed,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Same style txt/xlsx report as main.py, kept per-combo ---
    txt_path = os.path.join(save_dir, f"{algo_name}_{bit}_{problem_idx}.txt")
    xlsx_path = os.path.join(save_dir, f"{algo_name}_{bit}_{problem_idx}.xlsx")

    with open(txt_path, "w", encoding="utf-16") as f:
        f.write("========================================\n")
        f.write("      FINAL EXPERIMENTAL STATISTICS      \n")
        f.write("========================================\n")
        f.write(f"Problem: {problem_name} ({bit}-bit)\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Best Gate Counts per Trial: {best_scores_per_experiment}\n")
        f.write(f"Global Minimum Gate Count:  {global_min_gate}\n")
        f.write(f"Mean +/- Std (final best):  {mean_best:.4f} +/- {std_best:.4f}\n")
        f.write(f"Final Result (Gen {max_iterations}): {average_convergence_curve[-1]:.2f} +/- {std_convergence_curve[-1]:.2f}\n")
        f.write(f"Average Time per Experiment: {np.mean(execution_times):.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Circuit Structure:\n{absolute_best_circuit}\n")
        f.write("========================================\n")

    try:
        df = pd.DataFrame(fitness_history_matrix)
        df.index = [f"Trial_{i+1}" for i in range(df.shape[0])]
        df.columns = [f"Gen_{i+1}" for i in range(df.shape[1])]
        df["Execution_Time(s)"] = execution_times
        fitness_cols = [c for c in df.columns if c.startswith("Gen_")]
        stats_view = df[fitness_cols]
        df.loc["Average_Convergence"] = stats_view.mean()
        df.loc["Std_Deviation"] = stats_view.std()
        trial_execution_times = df["Execution_Time(s)"].iloc[:-2]
        df.at["Average_Convergence", "Execution_Time(s)"] = trial_execution_times.mean()
        df.to_excel(xlsx_path)
    except Exception as e:
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path)
        summary["xlsx_error"] = str(e)

    _write_progress(progress_path, {
        "bit": bit, "problem": problem_name, "algo": algo_name,
        "completed": num_experiments, "total": num_experiments, "status": "done",
        "start_time": start_time, "last_update": time.time(),
    })

    return summary
