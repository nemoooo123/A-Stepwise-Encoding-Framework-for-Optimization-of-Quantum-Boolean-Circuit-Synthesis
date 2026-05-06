from utils.init_state import gen_nbrs
from utils.topology import decode_and_synthesize,verify_circuit_logic
import matplotlib.pyplot as plt
import numpy as np
def plot_comprehensive_convergence(test1, test2, test3, test4, test5):
    """
    AE-QTS 演算法全指標收斂分析圖。
    
    Args:
        test1: global_worst_gate_count (歷史最差 GC 紀錄)
        test2: local_best_gate_count (當代最佳 GC)
        test3: local_best_continuity (最佳解路徑連續性成本)
        test4: local_best_composite (最佳解綜合代價分數)
        test5: 額外指標 (例如：Average Population Fitness 或 Diversity)
    """
    generations = range(1, len(test1) + 1)
    
    # 建立三子圖佈局 (3 rows, 1 column)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # --- 子圖 1: 閘數邊界分析 (Gate Count Boundary Analysis) ---
    ax1.plot(generations, test1, label='Global Worst GC (Upper Bound)', color='red', linestyle='--', alpha=0.5)
    ax1.plot(generations, test2, label='Local Best GC', color='blue', linewidth=2)
    ax1.set_ylabel('Gate Count', fontsize=12, fontweight='bold')
    ax1.set_title('Gate Count Convergence and Upper Bound Tracking', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 子圖 2: 綜合適應度分析 (Multi-Objective Cost Analysis) ---
    ax2.plot(generations, test3, label='Path Continuity Cost', color='green', alpha=0.7)
    ax2.plot(generations, test4, label='Composite Fitness Cost', color='purple', linewidth=2)
    ax2.set_ylabel('Cost Score', fontsize=12, fontweight='bold')
    ax2.set_title('Evolutionary Trajectories of Cost Metrics', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- 子圖 3: 額外監控指標 (Supplementary Metric Analysis) ---
    # 假設 test5 是平均值或其他指標
    ax3.plot(generations, test5, label='Population Metric / Avg Fitness', color='orange', linewidth=1.5)
    ax3.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax3.set_title('Dynamic Analysis of Supplementary Evolutionary Metric', fontsize=13)
    ax3.legend(loc='best')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_final_test_666(test_666):
    """
    專為 10,000 代設計的 test_666 統計演化圖
    """
    data = np.array(test_666)
    
    # 確保數據維度正確 (Generations, Samples)
    if data.ndim == 1:
        generations = np.arange(1, len(data) + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(generations, data, color='#1f77b4', linewidth=0.8, alpha=0.8)
    else:
        generations = np.arange(1, data.shape[0] + 1)
        avg_val = np.mean(data, axis=1)
        max_val = np.max(data, axis=1)
        min_val = np.min(data, axis=1)
        
        plt.figure(figsize=(12, 6))
        # 藍色陰影：族群分佈範圍（Max-Min）
        plt.fill_between(generations, min_val, max_val, color='#1f77b4', alpha=0.2, label='Population Spread')
        # 藍色主線：族群平均值
        plt.plot(generations, avg_val, color='#1f77b4', linewidth=1, label='Population Mean')
        # 紅色細線：當代最強解（這是衝擊 185 閘的關鍵指標）
        plt.plot(generations, min_val, color='red', linewidth=0.5, alpha=0.6, label='Current Best Candidate')

    plt.title('Statistical Evolution of test_666 Over 10,000 Generations', fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    # 避免 X 軸科學計數法太擠
    plt.ticklabel_format(style='plain', axis='x') 
    
    plt.tight_layout()
    plt.show()
def AE_QTS_run_single_experiment(max_iterations, 
                                 rotation_cycles, 
                                 num_neighbors, 
                                 num_bits, 
                                 base_trajectory, 
                                 experiment_id, 
                                 encoding_table, 
                                 qindividuals1, 
                                 qindividuals2, 
                                 qindividuals3, 
                                 qindividuals4, 
                                 fitness_history_matrix, 
                                 target_output,
                                 delta_theta):
    """
    Executes a single trial of the AE-QTS algorithm.
    Iteratively updates quantum individuals (qindividuals1-4) to minimize circuit gate count.
    """
    
    num_cycles = len(rotation_cycles)
    current_iter = 0
    global_best_gate_count = float('inf')
    global_worst_gate_count = float('-inf')
    global_best_circuit = []
    test1=[] #global_worst_gate_count 全域最差解
    test2=[] #local_best_gate_count  #當代最佳解
    test3=[] #最佳解指標1數值
    test4=[] #最佳解總數直
    test5=[] #最佳解指標2
    test_666=[] #看指標一數量

    while current_iter < max_iterations:
        current_iter += 1
        
        # Step 1: Neighborhood Generation
        # Create candidate solutions (neighbors) based on the current quantum state of qindividuals
        nbr1, nbr2, nbr3, nbr4 = gen_nbrs(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, num_neighbors
        )

        # Step 2: Decoding and Circuit Synthesis
        # Convert quantum neighbors into concrete reversible circuit solutions
        circuit_solutions, global_worst_gate_count, test_666 = decode_and_synthesize(
            nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory, global_worst_gate_count, global_best_gate_count, test_666
        )
        # Step 3: Fitness Evaluation (Gate Count Analysis)
        # Pair each solution's gate count with its original neighborhood index
        # Lower scores indicate superior individuals in our minimization framework.
        test1.append(global_worst_gate_count)

        solution_metrics = [(sol_tuple[3], idx) for idx, sol_tuple in enumerate(circuit_solutions)]
        # Sort by gate count in ascending order to identify the local optimal neighbor
        sorted_metrics = sorted(solution_metrics, key=lambda x: x[0])
        local_best_idx = sorted_metrics[0][1]
        local_best_gate_count = circuit_solutions[local_best_idx][1]
        local_best_circuit = circuit_solutions[local_best_idx][0]
        test2.append(local_best_gate_count)
        test5.append(local_best_gate_count/global_worst_gate_count)
        test3.append(circuit_solutions[local_best_idx][2])
        test4.append(circuit_solutions[local_best_idx][3])
        # Step 4: Quantum Population Update (Angle Adjustment)
        # Update the probability amplitudes of qindividuals based on neighbor performance
        updateQ(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
            num_neighbors, nbr1, nbr2, nbr3, nbr4, 
            [m[1] for m in sorted_metrics], num_cycles, delta_theta
        )

        # Step 5: Global Best Tracking
        # Update the overall best solution if a new minimum gate count is discovered
        if global_best_gate_count > local_best_gate_count:
            global_best_gate_count = local_best_gate_count
            global_best_circuit = local_best_circuit
        # Step 6: Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in circuit_solutions)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # Step 7: Data Recording for Statistical Analysis
        # Record the current best gate count into the fitness history matrix (used for np.mean later)
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count
    print(test_666)
    print(len(test_666))
    # plot_comprehensive_convergence(test1, test2, test3, test4, test5)   
    plot_final_test_666(test_666)
    return fitness_history_matrix, global_best_gate_count, global_best_circuit

def updateQ(qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
                               num_neighbors, nbr1, nbr2, nbr3, nbr4, 
                               sorted_indices, num_cycles, delta_theta):
    """
    Updates the probability distributions of quantum populations (q1-q4) based on 
    the relative fitness of their neighbors. 
    Uses a 'best-vs-worst' strategy to shift probability amplitudes.
    """
    
    # Process the top half of the population (Best half pulls away from Worst half)
    t = 0
    limit = int(num_neighbors / 2)
    
    while t < limit:
        # Step 1: Identify Best and Worst neighbor pairs
        # best_idx: index of the t-th best neighbor
        # worst_idx: index of the t-th worst neighbor
        best_idx = sorted_indices[t]
        worst_idx = sorted_indices[num_neighbors - 1 - t]

        # --- Update qindividuals1 (Strategy/Trajectory level) ---
        # best_sol1 and worst_sol1 represent the discrete decisions made by the neighbors
        best_sol1 = [list(map(int, row)) for row in nbr1[best_idx].tolist()]
        worst_sol1 = [list(map(int, row)) for row in nbr1[worst_idx].tolist()]

        for i in range(len(qindividuals1)):
            for j in range(len(qindividuals1[i])):
                b_val = best_sol1[i][j]
                w_val = worst_sol1[i][j]
                if b_val != w_val:
                    # Increment probability of the 'best' decision, decrement the 'worst'
                    qindividuals1[i][j][b_val] += delta_theta
                    qindividuals1[i][j][w_val] -= delta_theta
                    
                    # Boundary Correction: Ensure probabilities stay within [0, 1]
                    if qindividuals1[i][j][w_val] <= 0:
                        qindividuals1[i][j][b_val] = 1.0
                        qindividuals1[i][j][w_val] = 0.0

        # --- Update qindividuals2 (Segment level) ---
        best_sol2 = nbr2[best_idx]
        worst_sol2 = nbr2[worst_idx]

        for i in range(len(qindividuals2)):
            for j in range(len(qindividuals2[i])):
                if best_sol2[i][j] != worst_sol2[i][j]:
                    qindividuals2[i][j][best_sol2[i][j]] += delta_theta
                    qindividuals2[i][j][worst_sol2[i][j]] -= delta_theta
                    if qindividuals2[i][j][worst_sol2[i][j]] <= 0:
                        qindividuals2[i][j][best_sol2[i][j]] = 1.0
                        qindividuals2[i][j][worst_sol2[i][j]] = 0.0

        # --- Update qindividuals3 (Route/Path level) ---
        best_sol3 = nbr3[best_idx]
        worst_sol3 = nbr3[worst_idx]

        for i in range(len(qindividuals3)):
            for j in range(len(best_sol3[i])):
                for k in range(len(best_sol3[i][j])):
                    if best_sol3[i][j][k] != 999 and worst_sol3[i][j][k] != 999:
                        num_choices = len(qindividuals3[i][j][k])
                        for l in range(num_choices):
                            b_bit = best_sol3[i][j][k][l]
                            w_bit = worst_sol3[i][j][k][l]
                            if b_bit != w_bit:
                                qindividuals3[i][j][k][l][b_bit] += delta_theta
                                qindividuals3[i][j][k][l][w_bit] -= delta_theta
                                if qindividuals3[i][j][k][l][w_bit] <= 0:
                                    qindividuals3[i][j][k][l][b_bit] = 1.0
                                    qindividuals3[i][j][k][l][w_bit] = 0.0

        # --- Update qindividuals4 (Gate Order level) ---
        best_sol4 = nbr4[best_idx]
        worst_sol4 = nbr4[worst_idx]

        for i in range(num_cycles):
            for j in range(len(best_sol4[i])):
                # Only update if the gate sequence has more than one bit (non-trivial)
                if len(best_sol4[i][j]) > 1:
                    for k in range(len(best_sol4[i][j])):
                        b_gate = best_sol4[i][j][k]
                        w_gate = worst_sol4[i][j][k]
                        if b_gate != w_gate:
                            qindividuals4[i][j][k][b_gate] += delta_theta
                            qindividuals4[i][j][k][w_gate] -= delta_theta
                            if qindividuals4[i][j][k][w_gate] <= 0:
                                qindividuals4[i][j][k][b_gate] = 1.0
                                qindividuals4[i][j][k][w_gate] = 0.0
        t += 1

          
   