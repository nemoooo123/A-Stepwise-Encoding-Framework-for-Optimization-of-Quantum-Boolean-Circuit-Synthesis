import random
import copy
import numpy as np
from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

def WOA_run_single_experiment(max_iterations, rotation_cycles, num_neighbors, num_bits,
                                base_trajectory, experiment_id, encoding_table,
                                pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4,
                                fitness_history_matrix, target_output, b):
    
    # --- Step 1: 初始化族群 (鯨魚群) ---
    # 這裡產生的 nbr1~4 是當前所有鯨魚的位置
    nbr1, nbr2, nbr3, nbr4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors)
    
    # --- Step 2: 初始評估 ---
    circuit_solutions = decode_and_synthesize(nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory)
    fitness = [len(sol) for sol in circuit_solutions]
    current_circuits = list(circuit_solutions)
    
    # 紀錄全域領頭鯨魚 (Leader / Best Whale)
    best_idx = np.argmin(fitness)
    global_best_gate = fitness[best_idx]
    global_best_circuit = current_circuits[best_idx]
    leader_pos = (copy.deepcopy(nbr1[best_idx]), copy.deepcopy(nbr2[best_idx]), 
                  copy.deepcopy(nbr3[best_idx]), copy.deepcopy(nbr4[best_idx]))

    # --- Step 3: 進化迴圈 ---
    current_iter = 0
    while current_iter < max_iterations:
        current_iter += 1
        
        # 線性收斂因子 a，從 2 降到 0，控制 A 的搜索範圍
        a = 2 - current_iter * (2 / max_iterations)
        # valid_count=0
        for i in range(num_neighbors):
            p = random.random() # 決定執行 氣泡網(螺旋) 或 包圍/搜索 策略
            r = random.random()
            
            # WOA 核心參數 A 與 C
            # A 決定是要 向領頭移動(|A|<1) 還是 隨機探索(|A|>=1)
            # C 模擬獵物的隨機影響 (Influence)
            A = 2 * a * r - a
            C = 2 * r 
            
            if p < 0.5:
                if abs(A) < 1:
                    # [包圍階段] 向 Leader 靠近
                    new_child = generate_woa_update((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), leader_pos, C)
                else:
                    # [搜索階段] 隨機選一個鄰居靠近
                    rand_idx = random.choice([idx for idx in range(num_neighbors) if idx != i])
                    rand_pos = (nbr1[rand_idx], nbr2[rand_idx], nbr3[rand_idx], nbr4[rand_idx])
                    new_child = generate_woa_update((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), rand_pos, C)
            else:
                # [氣泡網階段] 螺旋更新 (針對 Leader 進行螺旋擾動)
                # 這裡傳入 b (螺旋常數) 來控制擾動劇烈程度
                new_child = generate_woa_spiral((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), leader_pos, b)

            # --- 評估新產生的解 ---
            new_per_circuit, new_fit = evaluate_woa_fitness(new_child, encoding_table, num_bits, base_trajectory)

            # --- 貪婪更新 (Greedy Selection) ---
            if new_fit < fitness[i]:
                nbr1[i], nbr2[i], nbr3[i], nbr4[i] = new_child
                fitness[i] = new_fit
                current_circuits[i] = new_per_circuit
                
                # 同步更新全域領頭鯨魚
                if new_fit < global_best_gate:
                    global_best_gate = new_fit
                    global_best_circuit = new_per_circuit
                    leader_pos = (copy.deepcopy(nbr1[i]), copy.deepcopy(nbr2[i]), 
                                  copy.deepcopy(nbr3[i]), copy.deepcopy(nbr4[i]))
                
            # Integrity Verification
            # Check if the synthesized circuits fulfill the logic requirements for the target output
            # valid_per_count = verify_circuit_logic(new_per_circuit, num_bits, target_output) 
            
        #     valid_count+=valid_per_count
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # 紀錄每一代的歷史最佳 Gate Count
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate

    return fitness_history_matrix, global_best_gate, global_best_circuit

# --- 輔助函式：位置更新邏輯 ---

def generate_woa_update(p1, target, C):
    """ 
    模擬鯨魚位置移動：
    利用參數 C 決定更新層數的深度，並執行 Crossover
    """
    child = list(copy.deepcopy(p1))
    
    # C 越大代表獵物吸引力越強，更新層數越多 (1~3層)
    num_layers = 1 if C < 0.6 else (2 if C < 1.4 else 3)
    layers = random.sample(range(4), num_layers)
    
    for l_idx in layers:
        curr_l = p1[l_idx]
        targ_l = target[l_idx]
        
        # 維度保護：確保內容為 List 格式以進行拼接
        c_list = curr_l.tolist() if isinstance(curr_l, np.ndarray) else list(curr_l)
        t_list = targ_l.tolist() if isinstance(targ_l, np.ndarray) else list(targ_l)
        
        if len(c_list) > 0:
            # 隨機選擇切點
            cp = random.randint(0, len(c_list))
            new_l = copy.deepcopy(c_list[:cp]) + copy.deepcopy(t_list[cp:])
            
            # Layer 4 專屬邏輯修復 (Sequence Order)
            if l_idx == 3:
                for k in range(len(new_l)):
                    for l in range(len(new_l[k])):
                        new_l[k][l] = repair_sequence_logic(new_l[k][l])
            child[l_idx] = new_l
            
    return tuple(child)

def generate_woa_spiral(p1, leader, b):
    """ 
    螺旋更新：
    以 Leader 為中心進行局部突變，模擬螺旋上升的氣泡網攻擊路徑
    參數 b 用於縮放突變機率 (Mutation Rate)
    """
    child = list(copy.deepcopy(leader))
    # 基礎突變率 5%，受 b 影響
    mutation_rate = 0.05 * b 
    
    for l_idx in range(4): # 0, 1, 2, 3 全部跑一遍
        layer = child[l_idx]
        
        # --- Layer 1 & 2 (2D 結構) ---
        if l_idx in [0, 1]:
            for r in range(len(layer)):
                for c in range(len(layer[r])):
                    if random.random() < mutation_rate:
                        layer[r][c] = 1 - layer[r][c]
                        
        # --- Layer 3 (深層 4D/5D 權重結構) ---
        elif l_idx == 2:
            # 這裡要小心處理 999 佔位符
            for j in range(len(layer)):
                for k in range(len(layer[j])):
                    for l in range(len(layer[j][k])):
                        # 只有當它不是 999 時才擾動
                        if not isinstance(layer[j][k][l], (int, float)) or layer[j][k][l] != 999:
                            if isinstance(layer[j][k][l], list):
                                for m in range(len(layer[j][k][l])):
                                    if random.random() < mutation_rate:
                                        layer[j][k][l][m] = 1 - layer[j][k][l][m]
        
        # --- Layer 4 (3D/4D 序列結構 + 雙層修復) ---
        elif l_idx == 3:
            for j in range(len(layer)):
                for k in range(len(layer[j])):
                    # 進入到序列層
                    for m in range(len(layer[j][k])):
                        if random.random() < mutation_rate:
                            layer[j][k][m] = 1 - layer[j][k][m]
                    # 擾動完立即修復該序列
                    layer[j][k] = repair_sequence_logic(layer[j][k])
                    
    return tuple(child)

def evaluate_woa_fitness(child, encoding_table, num_bits, trajectories):
    """ 封裝解碼與評估邏輯，並確保回傳電路內容與 Gate Count """
    c1, c2, c3, c4 = child
    # 傳入單一鯨魚位置進行解碼 (包裝成長度為 1 的族群)
    res = decode_and_synthesize([c1], [c2], [c3], [c4], encoding_table, num_bits, 1, trajectories)
    return res[0], len(res[0])