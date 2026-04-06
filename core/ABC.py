import random
import copy
import numpy as np
from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

# --- 核心實驗執行 ---
def ABC_run_single_experiment(max_iterations,
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
                limit):
    
    # 初始化族群 (蜜源)
    nbr1, nbr2, nbr3, nbr4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors)
    
    # 初始評估
    circuit_solutions = decode_and_synthesize(nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory)
    fitness = [len(sol) for sol in circuit_solutions]
    
    trials = [0] * num_neighbors # 紀錄蜜源沒進步的次數
    global_best_gate = min(fitness)
    global_best_circuit = circuit_solutions[np.argmin(fitness)]
    
    current_iter = 0
    while current_iter < max_iterations:
        current_iter += 1
        new_circuit = []
        # 1. 雇傭蜂階段: 每個蜜源都找鄰居比較
        nbr1, nbr2, nbr3, nbr4, fitness, trials, new_circuit = employed_bee_phase(
            nbr1, nbr2, nbr3, nbr4, fitness, new_circuit, trials, encoding_table, num_bits, base_trajectory)
        
        # 2. 觀察蜂階段: 根據品質（機率）加強搜尋
        nbr1, nbr2, nbr3, nbr4, fitness, trials, new_circuit = onlooker_bee_phase(
            nbr1, nbr2, nbr3, nbr4, fitness, new_circuit, trials, encoding_table, num_bits, base_trajectory)
        
        # 3. 偵查蜂階段: 拋棄太久沒進步的蜜源
        nbr1, nbr2, nbr3, nbr4, fitness, trials, new_circuit = scout_bee_phase(
            nbr1, nbr2, nbr3, nbr4, fitness, new_circuit, trials, limit, encoding_table, num_bits, base_trajectory)
        
        # Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in new_circuit)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")
        # 更新全域最優
        if min(fitness) < global_best_gate:
            global_best_gate = min(fitness)
            best_idx = np.argmin(fitness)
            # 重新合成最強電路
            global_best_circuit = new_circuit[best_idx]
        
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate

    return fitness_history_matrix, global_best_gate, global_best_circuit

# --- 1. 雇傭蜂階段 ---
def employed_bee_phase(nbr1, nbr2, nbr3, nbr4, fitness, new_circuit, trials, encoding_table, num_bits, trajectories):
    num_bees = int(len(fitness)/2)
    for i in range(num_bees):
        # 挑選一個鄰居 j
        j = random.choice([idx for idx in range(num_bees) if idx != i])
        
        # 產生新解 (針對某一層進行局部搜尋)
        new_child = generate_abc_neighbor( (nbr1[i], nbr2[i], nbr3[i], nbr4[i]), 
                                          (nbr1[j], nbr2[j], nbr3[j], nbr4[j]), num_bits )
        
        # 評估與貪婪選擇
        new_per_circuit, new_fit = evaluate_abc_fitness(new_child, encoding_table, num_bits, trajectories)
        new_circuit.append(new_per_circuit)

        if new_fit < fitness[i]:
            nbr1[i], nbr2[i], nbr3[i], nbr4[i] = new_child
            fitness[i] = new_fit
            trials[i] = 0
        else:
            trials[i] += 1
    return nbr1, nbr2, nbr3, nbr4, fitness, trials, new_circuit

# --- 2. 觀察蜂階段 ---
def onlooker_bee_phase(nbr1, nbr2, nbr3, nbr4, fitness, new_circuit, trials, encoding_table, num_bits, trajectories):
    
    num_bees = int(len(fitness)/2)
    # 計算機率 (與品質成正比，Gate Count 越低機率越高)
    # 為避免除以零，使用轉換公式
    weights = [1.0 / (f + 1e-6) for f in fitness]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    
    onlookers_dispatched = 0
    while onlookers_dispatched < num_bees:
        # 輪盤法選擇蜜源
        i = np.random.choice(range(len(fitness)), p=probs)
        j = random.choice([idx for idx in range(len(fitness)) if idx != i])
        
        new_child = generate_abc_neighbor( (nbr1[i], nbr2[i], nbr3[i], nbr4[i]), 
                                          (nbr1[j], nbr2[j], nbr3[j], nbr4[j]), num_bits )
        
        new_per_circuit, new_fit = evaluate_abc_fitness(new_child, encoding_table, num_bits, trajectories)
        new_circuit.append(new_per_circuit)

        if new_fit < fitness[i]:
            nbr1[i], nbr2[i], nbr3[i], nbr4[i] = new_child
            fitness[i] = new_fit
            trials[i] = 0
        else:
            trials[i] += 1
        onlookers_dispatched += 1
        
    return nbr1, nbr2, nbr3, nbr4, fitness, trials, new_circuit

# --- 3. 偵查蜂階段 ---
def scout_bee_phase(nbr1, nbr2, nbr3, nbr4, fitness, new_circuit, trials, limit, encoding_table, num_bits, trajectories):
    for i in range(len(trials)):
        if trials[i] > limit:
            # 放棄該蜜源，隨機重新生成 (Scout)
            # 這裡借用 gen_nbrs 邏輯生成 1 個新鄰居
            # 為了簡化，隨機修改現有解的多個位元來達成「隨機跳躍」
            for layer in [nbr1[i], nbr2[i]]:
                for r in range(len(layer)):
                    for c in range(len(layer[r])):
                        if random.random() < 0.3: # 大機率突變
                            layer[r][c] = 1 - layer[r][c]

            # 2. 處理 Layer 3 (4D 結構: [j][k][l][m])
            # 補上這段，偵查蜂才會真正「重生」
            for j in range(len(nbr3[i])):
                for k in range(len(nbr3[i][j])):
                    for l in range(len(nbr3[i][j][k])):
                        # 檢查是否為 999 佔位符
                        
                        if nbr3[i][j][k][l] != 999:
                            for m in range(len(nbr3[i][j][k][l])):
                                if random.random() < 0.3:
                                    nbr3[i][j][k][l][m] = 1 - nbr3[i][j][k][l][m]

            for j in range(len(nbr4[i])):
                for k in range(len(nbr4[i][j])):
                    for l in range(len(nbr4[i][j][k])):
                        if random.random() < 0.3:
                            nbr4[i][j][k][l] = 1 - nbr4[i][j][k][l]
                    nbr4[i][j][k] = repair_sequence_logic(nbr4[i][j][k])

                    

            # 重新評估
            new_per_circuit, fitness[i] = evaluate_abc_fitness((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), encoding_table, num_bits, trajectories)
            new_circuit[i] = new_per_circuit
            trials[i] = 0
    return nbr1, nbr2, nbr3, nbr4, fitness, trials, new_circuit

# --- 輔助函式: 產生鄰居解 (ABC 核心算子) ---
def generate_abc_neighbor(p1, p2, num_bits):
    """
    ABC 的局部搜尋：在爸爸的基礎上，隨機挑一層與鄰居進行「部分交換」
    """
    child = list(copy.deepcopy(p1))
    target_layer = random.randint(0, 3) # 隨機挑一層 1~4
    
    # 執行單點切換
    layer_p1 = p1[target_layer]
    layer_p2 = p2[target_layer]
    
    if len(layer_p1) > 0:
        l1_list = layer_p1.tolist() if isinstance(layer_p1, np.ndarray) else list(layer_p1)
        l2_list = layer_p2.tolist() if isinstance(layer_p2, np.ndarray) else list(layer_p2)
        cp = random.randint(0, len(l1_list))
        # 結構化合併
        new_l = copy.deepcopy(l1_list[:cp]) + copy.deepcopy(l2_list[cp:])
        
        # 如果是 Layer 4 記得修復
        if target_layer == 3:
            for k in range(len(new_l)):
                new_l[k] = repair_sequence_logic(new_l[k])
        
        child[target_layer] = new_l
        
    return tuple(child)

def evaluate_abc_fitness(child, encoding_table, num_bits, trajectories):
    c1, c2, c3, c4 = child
    circuit_sols = decode_and_synthesize([c1], [c2], [c3], [c4], encoding_table, num_bits, 1, trajectories)
    return circuit_sols[0], len(circuit_sols[0])

