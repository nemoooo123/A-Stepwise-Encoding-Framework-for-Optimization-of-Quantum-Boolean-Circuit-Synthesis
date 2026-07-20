import numpy as np
import random
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
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
        algo_choice = int(input("Select Algorithm (1: AE-QTS, 2: QTS, 3: QEA, 4: GA, 5: DE, 6: TS, 7: PSO, 8: WOA, 9: ABC): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Load the target truth table (output)
    target_output = loader.get_output(num_bits, problem_idx)
    if not target_output:
        print("Error: Targeted problem not found. Please restart.")
        return
    
    print(f"\nTarget Truth Table: {target_output}")

    # Fix the random seed so experiments are reproducible across runs
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Experimental Configuration
    num_experiments = 1
    max_iterations = 1000
    num_neighbors = 10  # Population size (N)
    
    print(f"Starting {num_experiments} experiments, each with {max_iterations} iterations...")

    # Pre-allocate matrix for convergence analysis (Rows: Experiments, Cols: Generations)
    # Using float('inf') as default to easily track minimization progress
    fitness_history_matrix = np.full((num_experiments, max_iterations), float('inf'))
    unique_history_matrix = np.full((num_experiments, max_iterations), float('inf'))
    a1_history_matrix = np.full((num_experiments, max_iterations), float('inf'))
    a2_history_matrix = np.full((num_experiments, max_iterations), float('inf'))
    best_scores_per_experiment = []
    best_circuit_per_experiment = []
    execution_times = []

    
    for r in range(num_experiments):
        # Track execution time for the current experiment trial
        experiment_start_time = time.time()
        # Step 1: Analyze permutation cycles in the truth table
        # check_zero_gate: Boolean to verify if the circuit requires any gates at all
        cycles, is_gate_required = find_cycles(target_output, check_zero_gate=True)
        
        if not is_gate_required:
            print(f"Experiment {r+1}: Identity mapping detected (Zero gates required).")
            best_scores_per_experiment.append(0)
            fitness_history_matrix[r, :] = 0
            unique_history_matrix[r, :] = 0
            a1_history_matrix[r, :] = 0
            a2_history_matrix[r, :] = 0
            continue

        if algo_choice == 1:  # AE-QTS (Amplitude-Ensemble Quantum-Inspired Tabu Search)
            # Step 2: Initialize Quantum Individuals and Encoding Tables
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, encoding_table, trajectory_base = build_encode(cycles)
            
            # Step 3: Run the core search algorithm for a single experiment
            fitness_history_matrix, unique_history_matrix, a1_history_matrix, a2_history_matrix, final_best_gate, best_circuit_this_run = AE_QTS_run_single_experiment(
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
                unique_history_matrix = unique_history_matrix,
                a1_history_matrix = a1_history_matrix,
                a2_history_matrix = a2_history_matrix,
                target_output = target_output,
                delta_theta = 0.01
            )

        elif algo_choice == 2: # QTS (Quantum-Inspired Tabu Search)
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
                target_output = target_output,
                delta_theta = 0.01
            )

        elif algo_choice == 3: # QEA (Quantum Evolutionary Algorithm - Global Best Guided)
            # Step 1: Initialize Quantum-Inspired Encoding Framework
            # Generates probability-based individuals (qindividuals) and discrete encoding tables.
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, encoding_table, trajectory_base = build_encode(cycles)
            
            # Step 2: Execute QEA Core Evolution
            # This algorithm uses a 'Global Best' oriented rotation strategy to shift
            # quantum probability amplitudes toward the historical optimal circuit configuration.
            fitness_history_matrix, final_best_gate, best_circuit_this_run = QEA_run_single_experiment(
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
                target_output = target_output,
                delta_theta = 0.002  # Rotation step size for QEA state updates
            )

        elif algo_choice == 4: # GA (Genetic Algorithm)

            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            fitness_history_matrix, final_best_gate, best_circuit_this_run = GA_run_single_experiment(
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
                k=3, 
                pc=0.82, # crossover
                pm=0.025 # mutation

            )
        elif algo_choice == 5: # DE (Differential Evolution - A population-based stochastic global optimizer)
            # Step 1: Initialize Encoding Fram
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            # Note: DE_run_single_experiment's own "fitness_history_matrix" param tracks the
            # unique-gate-count objective it searches on, while its "unique_history_matrix" param
            # carries the paired total gate count. Swap on unpack so main.py's fitness_history_matrix
            # always means "total" and unique_history_matrix always means "unique", matching AE-QTS.
            fitness_history_matrix, unique_history_matrix, a1_history_matrix, a2_history_matrix, final_best_gate, best_circuit_this_run = DE_run_single_experiment(
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
                unique_history_matrix = unique_history_matrix,
                a1_history_matrix = a1_history_matrix,
                a2_history_matrix = a2_history_matrix,
                target_output = target_output,
                CR = 0.05,
            )
        
        elif algo_choice == 6: # TS (Tubu Search) 
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
                tabu_size = 55
            )

        elif algo_choice == 7: # PSO (Particle Swarm Optimization)
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
                target_output = target_output,
                w=0.6,c1=3.0,c2=1.0
            )

        elif algo_choice == 8: # WOA (Whale Optimization Algorithm)
            # Step 1: Initialize Encoding Framework
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            fitness_history_matrix, final_best_gate, best_circuit_this_run = WOA_run_single_experiment(
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
                b = 1.2  # Spiral Constant
            )

        elif algo_choice == 9: # ABC(Artificial Bee Colony Algorithm)

            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, encoding_table, trajectory_base = build_encode(cycles)
            
            fitness_history_matrix, final_best_gate, best_circuit_this_run = ABC_run_single_experiment(
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
                limit = 25
            )

        

        experiment_end_time = time.time()
        elapsed_time = experiment_end_time - experiment_start_time
        execution_times.append(elapsed_time)
            
        best_circuit_per_experiment.append(best_circuit_this_run)
        best_scores_per_experiment.append(final_best_gate)

        # Progress reporting
        print(f"--- Experiment {r+1} Finished | Best Gate: {final_best_gate} | Time: {elapsed_time:.2f}s ---")
    # Statistical Analysis and Result Aggregation
    # Calculate the average gate count across all experiments for each generation
    average_convergence_curve = np.mean(fitness_history_matrix, axis=0)
    std_convergence_curve = np.std(fitness_history_matrix, axis=0)

    # Unique gate count convergence is only populated for algorithms that track it (AE-QTS, DE)
    has_unique_data = algo_choice in (1, 5)
    if has_unique_data:
        average_unique_curve = np.mean(unique_history_matrix, axis=0)

    # a1/a2 counts are only populated for AE-QTS and DE
    has_a1_a2_data = algo_choice in (1, 5)
    if has_a1_a2_data:
        average_a1_curve = np.mean(a1_history_matrix, axis=0)
        average_a2_curve = np.mean(a2_history_matrix, axis=0)


    global_min_gate = min(best_scores_per_experiment)
    best_exp_index = best_scores_per_experiment.index(global_min_gate) # Identify the most successful trial
    absolute_best_circuit = best_circuit_per_experiment[best_exp_index]
    total_end_time = time.time()
    
    final_avg = average_convergence_curve[-1]
    final_std = std_convergence_curve[-1]

# --- Automatic Directory and Filename Creation ---
    # Algorithm mapping: mapping choice ID to descriptive names
    algo_names = {
        1: "AE-QTS", 2: "QTS", 3: "QEA", 4: "GA", 5: "DE", 
        6: "TS", 7: "PSO", 8: "WOA", 9: "ABC"
    }
    algo_name = algo_names.get(algo_choice, "Other")
    
    # Establish directory path structure (e.g., exp/13_bit/QEA_Results/)
    save_dir = os.path.join("exp", f"{num_bits}_bit", f"{algo_name}_Results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define file path foundations
    base_filename = f"{algo_name}_{num_bits}_{problem_idx}"
    txt_path = os.path.join(save_dir, f"{base_filename}.txt")
    xlsx_path = os.path.join(save_dir, f"{base_filename}.xlsx")

    # --- Generate Averaged Convergence Plot (across all experiments) ---
    plt.figure()
    iterations = range(1, max_iterations + 1)
    plt.plot(iterations, average_convergence_curve, label='avg total gate count')
    if has_unique_data:
        plt.plot(iterations, average_unique_curve, label='avg unique gate count')
    # if has_a1_a2_data:
    #     plt.plot(iterations, average_a1_curve, label='avg a1 count')
    #     plt.plot(iterations, average_a2_curve, label='avg a2 count')
    plt.xlabel('Iteration')
    plt.ylabel('Gate Count')
    plt.title(f'{algo_name} Averaged Convergence Curve ({num_experiments} experiments)')
    plt.legend()
    avg_plot_path = os.path.join(save_dir, f"{base_filename}_avg_convergence.png")
    plt.savefig(avg_plot_path)
    plt.close()
    print(f"[System] Averaged convergence plot saved to: {avg_plot_path}")

    # --- Generate TXT Summary Report ---
    # This file provides a quick overview of the best results and system performance
    with open(txt_path, "w", encoding="utf-16") as f:
        f.write("========================================\n")
        f.write("      FINAL EXPERIMENTAL STATISTICS      \n")
        f.write("========================================\n")
        f.write(f"Best Gate Counts per Trial: {best_scores_per_experiment}\n")
        f.write(f"Global Minimum Gate Count:  {global_min_gate}\n")
        f.write(f"Final Result (Gen {max_iterations}): {final_avg:.2f} ± {final_std:.2f}\n")
        f.write(f"Average Time per Experiment: {sum(execution_times)/len(execution_times):.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Circuit Structure:\n{absolute_best_circuit}\n")
        f.write("========================================\n")

    # --- Generate Detailed Excel Data (Pandas) ---
    # This file stores the full convergence matrices (total/unique gate counts, a1, a2) for statistical plotting and verification

    def build_history_df(matrix):
        # 1. Construct the basic DataFrame (Rows: Trials, Columns: Generations)
        df = pd.DataFrame(matrix)
        df.index = [f"Trial_{i+1}" for i in range(df.shape[0])]
        df.columns = [f"Gen_{i+1}" for i in range(df.shape[1])]

        # 2. Append the Execution Time column
        # Ensure 'execution_times' is a list of floats/ints with the same length as num_experiments
        df['Execution_Time(s)'] = execution_times

        # 3. Calculate statistical rows (Mean and Std Deviation)
        # CRITICAL: We only calculate stats for 'Gen_x' columns, excluding 'Execution_Time(s)'
        gen_cols = [c for c in df.columns if c.startswith('Gen_')]

        # Create a temporary view for calculation
        stats_view = df[gen_cols]

        # Append Average and Std Dev rows
        df.loc['Average_Convergence'] = stats_view.mean()
        df.loc['Std_Deviation'] = stats_view.std()

        # 4. Manually fill the Execution Time average for the summary row
        # Calculate the mean execution time across all successful trials
        trial_execution_times = df['Execution_Time(s)'].iloc[:-2]
        df.at['Average_Convergence', 'Execution_Time(s)'] = trial_execution_times.mean()
        return df

    df = build_history_df(fitness_history_matrix)
    try:
        # EXPORT TO FILE: one sheet per tracked metric
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, sheet_name='Total_Gate_Count')
            if has_unique_data:
                build_history_df(unique_history_matrix).to_excel(writer, sheet_name='Unique_Gate_Count')
            if has_a1_a2_data:
                build_history_df(a1_history_matrix).to_excel(writer, sheet_name='A1_Count')
                build_history_df(a2_history_matrix).to_excel(writer, sheet_name='A2_Count')

        # Console Feedback
        print(f"\n[System] Summary report saved to: {txt_path}")
        print(f"[System] Full convergence history with execution times saved to: {xlsx_path}")

    except Exception as e:
        print(f"\n[Error] Failed to generate Excel file: {e}")
        # If Excel fails, try saving as CSV as a backup
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path)
        print(f"[Backup] Data saved to CSV instead: {csv_path}")

    


if __name__ == "__main__":
    main()
    