from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

import random
import copy
import numpy as np

def GA_run_single_experiment(
                max_iterations,
                rotation_cycles,
                num_neighbors,
                num_bits,
                base_trajectory,
                experiment_id,
                encoding_table,
                pop_matrix1,
                pop_matrix2,
                pop_matrix3,
                pop_matrix4,
                fitness_history_matrix,
                target_output,
                k,
                pc,
                pm
            ):
    
    # --- Step 1: 初始化族群 ---
    # 傳入的矩陣產生初始鄰居 
    nbr1, nbr2, nbr3, nbr4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors)
    
    # --- Step 2: 初始評估 ---
    circuit_solutions = decode_and_synthesize(
        nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory
    )
    
    # 計算初始 Fitness (Gate Count)
    # 格式: [gate_count, gate_count, ...]
    local_fitness = [len(sol) for sol in circuit_solutions]
    local_indices = list(range(num_neighbors)) # 初始索引 [0, 1, 2, ...]
    
    global_best_gate_count = min(local_fitness)
    best_idx = np.argmin(local_fitness)
    global_best_circuit = circuit_solutions[best_idx]
    local_best_circuit = circuit_solutions[best_idx]
    # --- Step 3: 進化迴圈 ---
    current_iter = 0
    while current_iter < max_iterations:
        current_iter += 1
        
        # 執行 GA 核心操作 (選擇、交配、突變、評估)
        # 這裡會回傳新一代的 矩陣 與 Fitness
        nbr1, nbr2, nbr3, nbr4, local_fitness, local_circuit = Genetic_Algorithm_Core(
            nbr1, nbr2, nbr3, nbr4, num_neighbors, num_bits, k, pc, pm,
            encoding_table, local_fitness, local_best_circuit, local_indices, base_trajectory
        )
        #Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in local_circuit)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")
        # --- Step 4: 更新全域最佳解 ---
        # 直接拿當前這一代最強者的「索引」
        current_local_best_idx = np.argmin(local_fitness)
        # 透過索引拿「數值」
        current_local_best_fit = local_fitness[current_local_best_idx]
        local_best_circuit = local_circuit[current_local_best_idx]
        if current_local_best_fit < global_best_gate_count:

            global_best_gate_count = current_local_best_fit
            global_best_circuit = local_best_circuit

        # 記錄歷史數據供繪圖使用
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count

    return fitness_history_matrix, global_best_gate_count, global_best_circuit


def Genetic_Algorithm_Core(nbr1, nbr2, nbr3, nbr4, num_neighbors, num_bits, k, pc, pm,
                           encoding_table, local_fitness, local_best_circuit, local_indices, base_trajectory):
    """
    執行單代的 GA 演化：精英保留 -> 選擇 -> 交配 -> 突變 -> 評估
    """
    new_nbr1, new_nbr2, new_nbr3, new_nbr4, new_fit, new_circuit = [], [], [], [], [], []

    # 1. 精英保留 (Elitism): 確保最強個體直接進入下一代
    best_in_current_idx = np.argmin(local_fitness)
    new_nbr1.append(copy.deepcopy(nbr1[best_in_current_idx]))
    new_nbr2.append(copy.deepcopy(nbr2[best_in_current_idx]))
    new_nbr3.append(copy.deepcopy(nbr3[best_in_current_idx]))
    new_nbr4.append(copy.deepcopy(nbr4[best_in_current_idx]))
    new_fit.append(local_fitness[best_in_current_idx])
    new_circuit.append(local_best_circuit)

    # 2. 產生其餘子代
    while len(new_fit) < num_neighbors:
        # 選擇父母 (錦標賽)
        p1_idx = tournament_selection(local_fitness, local_indices, k)
        p2_idx = tournament_selection(local_fitness, local_indices, k)
        
        p1 = (nbr1[p1_idx], nbr2[p1_idx], nbr3[p1_idx], nbr4[p1_idx])
        p2 = (nbr1[p2_idx], nbr2[p2_idx], nbr3[p2_idx], nbr4[p2_idx])

        # 交配
        if random.random() < pc:
            child = crossover_op(p1, p2)
        else:
            child = copy.deepcopy(p1)
        # 突變
        child = mutation_op(child, pm)

        c1, c2, c3, c4 = child
        # 評估新子代 只有一組
        
        circuit_solutions = decode_and_synthesize(
        [c1], [c2], [c3], [c4], encoding_table, num_bits, 1, base_trajectory
        )
        fit = len(circuit_solutions[0])
        
        new_nbr1.append(c1)
        new_nbr2.append(c2)
        new_nbr3.append(c3)
        new_nbr4.append(c4)
        new_circuit.append(circuit_solutions[0])
        new_fit.append(fit)

    return new_nbr1, new_nbr2, new_nbr3, new_nbr4, new_fit, new_circuit

def tournament_selection(fitness_list, idx_list, k):
    """
    從族群中隨機選出 k 個個體，回傳其中適應度最好（閘數最少）的索引。
    k 越大，選擇壓力越大，收斂越快。
    """
    # 隨機挑選 k 個參賽者的索引
    tournament_indices = random.sample(range(len(fitness_list)), k)
    
    # 找出這 k 個參賽者中 Fitness 最小（Gate Count 最少）的那個
    best_in_tournament = min(tournament_indices, key=lambda i: fitness_list[i])
    
    # 回傳該個體在原始族群中的位置
    return idx_list[best_in_tournament]

def crossover_op(p1, p2):
    """
    對四個編碼層級分別進行交配。
    p1, p2 分別是父母的 (nbr1, nbr2, nbr3, nbr4) 元組。
    """
    child = []
    # 對每一層編碼 (nbr1 到 nbr4) 分別做處理
    for layer_p1, layer_p2 in zip(p1, p2):
        # 確保有編碼內容才進行切點選擇
        if len(layer_p1) > 0:
            # 隨機選擇一個切點 (Crossover Point)
            cp = random.randint(0, len(layer_p1))
            # 先轉成 list 再相加，這樣不論原始格式是什麼都能正確合併
            l1_list = layer_p1.tolist() if isinstance(layer_p1, np.ndarray) else list(layer_p1)
            l2_list = layer_p2.tolist() if isinstance(layer_p2, np.ndarray) else list(layer_p2)
            
            new_layer = copy.deepcopy(l1_list[:cp]) + copy.deepcopy(l2_list[cp:])
        else:
            new_layer = copy.deepcopy(layer_p1)
        child.append(new_layer)
        
    return tuple(child) # 回傳 (c1, c2, c3, c4)

def mutation_op(child, pm):
    """
    對個體的每一位元以 pm 機率進行翻轉 (0變1, 1變0)。
    """
    c1, c2, c3, c4 = child

    # Layer 1 & 2 (2D)
    for layer in [c1, c2]:
        for i in range(len(layer)):
            for j in range(len(layer[i])):
                if random.random() < pm:
                    layer[i][j] = 1 - layer[i][j]

    # Layer 3 (4D: i, j, k, l)
    for i in range(len(c3)):
        for j in range(len(c3[i])):
            for k in range(len(c3[i][j])):
                if c3[i][j][k] != 999:
                    for l in range(len(c3[i][j][k])):
                        if random.random() < pm:
                            c3[i][j][k][l] = 1 - c3[i][j][k][l]

    # Layer 4 (3D: i, j, k)
    for i in range(len(c4)):
        for j in range(len(c4[i])):
            # 突變位元
            for k in range(len(c4[i][j])):
                if random.random() < pm:
                    c4[i][j][k] = 1 - c4[i][j][k]
            # 突變完立刻修復該序列
            c4[i][j] = repair_sequence_logic(c4[i][j])
        
    return (c1, c2, c3, c4)



