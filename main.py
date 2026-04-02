import numpy as np
import time
import os
from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode, gen_nbrs
from core.AE_QTS import AE_QTS_run_single_experiment
from core.DE import DE_run_single_experiment
from core.PSO import PSO_run_single_experiment
from core.TS import TS_run_single_experiment
from core.QTS import QTS_run_single_experiment
def main():
    """
    Main execution entry point for Reversible Circuit Synthesis experiments.
    Handles data loading, algorithm selection, and statistical aggregation.
    """
    # Initialize data loader and show available problem sets
    loader = DataLoader()
    loader.get_info()
    # User Input Section
    try:
        num_bits = int(input("Enter number of bits (n): "))
        problem_idx = int(input("Enter problem index: "))
        algo_choice = int(input("Select Algorithm (1: AE-QTS, 2: DE, 3: PSO, 4: TS, 5: QTS, 6: Other): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Load the target truth table (output)
    target_output = loader.get_output(num_bits, problem_idx)
    if not target_output:
        print("Error: Targeted problem not found. Please restart.")
        return
    
    print(f"\nTarget Truth Table: {target_output}")

    # Experimental Configuration
    num_experiments = 10
    max_iterations = 1000
    num_neighbors = 10  # Population size (N)
    
    print(f"Starting {num_experiments} experiments, each with {max_iterations} iterations...")

    # Pre-allocate matrix for convergence analysis (Rows: Experiments, Cols: Generations)
    # Using float('inf') as default to easily track minimization progress
    fitness_history_matrix = np.full((num_experiments, max_iterations), float('inf'))
    best_scores_per_experiment = []
    best_circuit_per_experiment = []
    execution_times = []

    
    for r in range(num_experiments):
        #time count
        experiment_start_time = time.time()
        # Step 1: Analyze permutation cycles in the truth table
        # check_zero_gate: Boolean to verify if the circuit requires any gates at all
        cycles, is_gate_required = find_cycles(target_output, check_zero_gate=True)
        
        if not is_gate_required:
            print(f"Experiment {r+1}: Identity mapping detected (Zero gates required).")
            best_scores_per_experiment.append(0)
            fitness_history_matrix[r, :] = 0
            continue

        if algo_choice == 1:  # AE-QTS (Amplitude-Ensemble Quantum-Inspired Tabu Search)
            # Step 2: Initialize Quantum Individuals and Encoding Tables
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, encoding_table, trajectory_base = build_encode(cycles)
            
            # Step 3: Run the core search algorithm for a single experiment
            fitness_history_matrix, final_best_gate, best_circuit_this_run = AE_QTS_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                qindividuals1 = qindividuals1,
                qindividuals2 = qindividuals2,
                qindividuals3 = qindividuals3,
                qindividuals4 = qindividuals4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output
            )
            

        elif algo_choice == 2: # DE (Differential Evolution - Classical Metaheuristic)
            # Step 1: Initialize Encoding Fram
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            fitness_history_matrix, final_best_gate , best_circuit_this_run = DE_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                CR = 0.01,
            )

        elif algo_choice == 3: # PSO (Particle Swarm Optimization)
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            fitness_history_matrix, final_best_gate, best_circuit_this_run = PSO_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output
            )
        
        elif algo_choice == 4: # TS (Tubu Search) 
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            fitness_history_matrix, final_best_gate, best_circuit_this_run = TS_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                tabu_size = 7
            )
        
        elif algo_choice == 5: # QTS (Quantum-Inspired Tabu Search)
            # Step 2: Initialize Quantum Individuals and Encoding Tables
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, encoding_table, trajectory_base = build_encode(cycles)
            
            # Step 3: Run the core search algorithm for a single experiment
            fitness_history_matrix, final_best_gate, best_circuit_this_run = QTS_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                qindividuals1 = qindividuals1,
                qindividuals2 = qindividuals2,
                qindividuals3 = qindividuals3,
                qindividuals4 = qindividuals4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output
            )

        experiment_end_time = time.time()
        elapsed_time = experiment_end_time - experiment_start_time
        execution_times.append(elapsed_time)
            
        best_circuit_per_experiment.append(best_circuit_this_run)
        best_scores_per_experiment.append(final_best_gate)

        # Progress reporting
        print(f"--- Experiment {r+1} Finished | Best Gate: {final_best_gate} | Time: {elapsed_time:.2f}s ---")
    # Step 4: Statistical Analysis and Result Aggregation
    # Calculate the average gate count across all experiments for each generation
    average_convergence_curve = np.mean(fitness_history_matrix, axis=0)
    std_convergence_curve = np.std(fitness_history_matrix, axis=0)
    
    global_min_gate = min(best_scores_per_experiment)
    best_exp_index = best_scores_per_experiment.index(global_min_gate)
    absolute_best_circuit = best_circuit_per_experiment[best_exp_index]
    total_end_time = time.time()

    print("\n" + "="*40)
    print("      FINAL EXPERIMENTAL STATISTICS      ")
    print("="*40)
    print(f"Full Fitness History Matrix (Raw Data):\n{fitness_history_matrix}")
    print("-" * 40)
    print(f"Best Gate Counts per Trial: {best_scores_per_experiment}")
    print(f"Global Minimum Gate Count:  {global_min_gate}")
    print(f"Best Circuit (from Exp {best_exp_index + 1}):\n{absolute_best_circuit}")
    print("-" * 40)
    print(f"Average Gates = {average_convergence_curve}")
    print(f"Std Deviation = {std_convergence_curve}")
    print(f"Best Circuit:  {best_circuit_per_experiment}")
    print("-" * 40)
    final_avg = average_convergence_curve[-1]
    final_std = std_convergence_curve[-1]
    print(f"Final Result (Gen {max_iterations}): {final_avg:.2f} ± {final_std:.2f}")

    print(f"Average Time per Experiment: {sum(execution_times)/len(execution_times):.2f}s")
    print("="*40)
    

    # --- 自動建立資料夾與檔名 ---
    # 假設 algo_choice 對應名稱：1: AE-QTS, 2: DE, 3: PSO, 4: TS, 5: QTS
    algo_names = {1: "AE-QTS", 2: "DE", 3: "PSO", 4: "TS", 5: "QTS"}
    algo_name = algo_names.get(algo_choice, "Other")
    
    # 建立目錄路徑 (例如: exp/3_bit/AE-QTS_Results/)
    save_dir = os.path.join("exp", f"{num_bits}_bit", f"{algo_name}_Results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 檔案名稱 (例如: AE-QTS_3_1.txt)
    save_path = os.path.join(save_dir, f"{algo_name}_{num_bits}_{problem_idx}.txt")

    # --- 寫入檔案 ---
    with open(save_path, "w", encoding="utf-16") as f:
        f.write("========================================\n")
        f.write("      FINAL EXPERIMENTAL STATISTICS      \n")
        f.write("========================================\n")
        f.write(f"Full Fitness History Matrix (Raw Data):\n{fitness_history_matrix}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Gate Counts per Trial: {best_scores_per_experiment}\n")
        f.write(f"Global Minimum Gate Count:  {global_min_gate}\n")
        f.write(f"Best Circuit (from Exp {best_exp_index + 1}):\n{absolute_best_circuit}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Gates = {average_convergence_curve}\n")
        f.write(f"Std Deviation = {std_convergence_curve}\n")
        f.write(f"Best Circuit:  {best_circuit_per_experiment}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Result (Gen {max_iterations}): {final_avg:.2f} ± {final_std:.2f}\n")
        f.write(f"Average Time per Experiment: {sum(execution_times)/len(execution_times):.2f}s\n")
        f.write("========================================\n")

    print(f"\n[系統] 實驗數據已成功存檔至: {save_path}")
    


if __name__ == "__main__":
    main()
    