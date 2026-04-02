import numpy as np
import copy
import random
from utils.init_state import gen_pso_init_states, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

def PSO_run_single_experiment(
    max_iterations, rotation_cycles, num_neighbors, num_bits, base_trajectory,
    experiment_id, encoding_table, pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4,
    fitness_history_matrix, target_output, w=0.8, c1=1.5, c2=1.5
):
    """
    Execute a single PSO experiment: Each neighbor is treated as a particle 
    with a continuous position and velocity vector across four encoding layers.
    """
    
    # --- Step 1: Initialization ---
    # Generate initial particle states (positions, velocities, and discrete solutions)
    (curr_nbr1, vel1, discrete_nbr1,
     curr_nbr2, vel2, discrete_nbr2,
     curr_nbr3, vel3, discrete_nbr3,
     curr_nbr4, vel4, discrete_nbr4) = gen_pso_init_states(
        pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors
    )

    # Initialize Personal Best (pBest) positions. 
    # Must record continuous values to maintain momentum in the velocity formula.
    pbest_n1, pbest_n2, pbest_n3, pbest_n4 = (
        copy.deepcopy(curr_nbr1), copy.deepcopy(curr_nbr2), 
        copy.deepcopy(curr_nbr3), copy.deepcopy(curr_nbr4)
    )
    
    
    # Decode and synthesize circuits to calculate initial population fitness
    circuit_solutions = decode_and_synthesize(
        discrete_nbr1, discrete_nbr2, discrete_nbr3, discrete_nbr4, 
        encoding_table, num_bits, num_neighbors, base_trajectory
    )
    # Define fitness as the gate count (Optimization goal: Minimize gate count)
    pbest_fitness = [len(sol) for sol in circuit_solutions]
    
    # Initialize Global Best (gBest)
    gbest_idx = np.argmin(pbest_fitness) # Find the index of the particle with the minimum gates
    gbest_gate_count = pbest_fitness[gbest_idx] # Record the minimum gate count found so far
    
    # Perform a deep copy of the global best position across all four layers
    gbest_n1, gbest_n2, gbest_n3, gbest_n4 = (
        copy.deepcopy(pbest_n1[gbest_idx]), copy.deepcopy(pbest_n2[gbest_idx]),
        copy.deepcopy(pbest_n3[gbest_idx]), copy.deepcopy(pbest_n4[gbest_idx])
    )
    gbest_circuit = circuit_solutions[gbest_idx] # Store the best circuit structure found
    
    current_iter = 0
    # --- Step 2: Swarm Evolution Loop ---
    while current_iter < max_iterations:
        current_iter += 1

        for i in range(num_neighbors):
            # 生成隨機因子 r1, r2 增加搜尋隨機性
            
            # Update Layer 1 (Global Selection): Sync Pos, Vel, and Discrete bits
            vel1[i], curr_nbr1[i], discrete_nbr1[i] = pso_recursive_sync_update(
                vel1[i], curr_nbr1[i], pbest_n1[i], gbest_n1, w, c1, c2 ,True
            )   

            # Update Layer 2 (Entry Points)
            vel2[i], curr_nbr2[i], discrete_nbr2[i] = pso_recursive_sync_update(
                vel2[i], curr_nbr2[i], pbest_n2[i], gbest_n2, w, c1, c2 ,True
            )   

            # Update Layer 3 (Intermediate Path Nodes)
            vel3[i], curr_nbr3[i], discrete_nbr3[i] = pso_recursive_sync_update(
                vel3[i], curr_nbr3[i], pbest_n3[i], gbest_n3, w, c1, c2 ,True
            )   

            # Update Layer 4 (Gate Sequencing): Convert 999 markers to 0 for sequence processing
            vel4[i], curr_nbr4[i], discrete_nbr4[i] = pso_recursive_sync_update(
                vel4[i], curr_nbr4[i], pbest_n4[i], gbest_n4, w, c1, c2 ,False
            )      
            # Post-Update Constraint Satisfaction: Repair gate sequences in Layer 4
            for cycle_idx in range(len(discrete_nbr4[i])):
                for step_idx in range(len(discrete_nbr4[i][cycle_idx])):
                    # Perform logic repair on the innermost bit sequence
                    repaired_bits = repair_sequence_logic(discrete_nbr4[i][cycle_idx][step_idx])
                    discrete_nbr4[i][cycle_idx][step_idx] = repaired_bits
            
        # Synthesize updated circuits and evaluate new fitness values for the current population
        circuit_solutions = decode_and_synthesize(
            discrete_nbr1, discrete_nbr2, discrete_nbr3, discrete_nbr4, 
            encoding_table, num_bits, num_neighbors, base_trajectory
        )
        # Identify local best of the current iteration (Iteration Best)
        lbest_fitness = [len(sol) for sol in circuit_solutions]

        # Identify local best of the current iteration (Iteration Best)
        lbest_idx = np.argmin(lbest_fitness) 
        lbest_gate_count = lbest_fitness[lbest_idx] 
        
        # --- Step 3: Update Personal Best (pBest) and Global Best (gBest) ---
        for i in range(num_neighbors):
            if lbest_fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = lbest_fitness[i]  # Update gBest fitness score
                # Update gBest position data (continuous coordinates)
                pbest_n1[i], pbest_n2[i], pbest_n3[i], pbest_n4[i] = (
                    copy.deepcopy(curr_nbr1[i]), copy.deepcopy(curr_nbr2[i]),
                    copy.deepcopy(curr_nbr3[i]), copy.deepcopy(curr_nbr4[i])
                )
            
        
        # If the best particle in this iteration outperforms the global best
        if lbest_gate_count < gbest_gate_count:
            gbest_gate_count = lbest_gate_count # Update gBest fitness score
            gbest_circuit = circuit_solutions[lbest_idx]
            # Update gBest position data (continuous coordinates)
            gbest_n1, gbest_n2, gbest_n3, gbest_n4 = (
                copy.deepcopy(curr_nbr1[lbest_idx]), copy.deepcopy(curr_nbr2[lbest_idx]),
                copy.deepcopy(curr_nbr3[lbest_idx]), copy.deepcopy(curr_nbr4[lbest_idx])
            )

        # Log the global best gate count for the current iteration into the history matrix
        fitness_history_matrix[experiment_id][current_iter - 1] = gbest_gate_count
        
    # Return fitness history, final global best gate count, and the optimized circuit structure
    return fitness_history_matrix, gbest_gate_count, gbest_circuit


def pso_recursive_sync_update(v, x, pbest, gbest, w, c1, c2, keep_999):
    """
    Recursively updates PSO velocity and position, and generates discrete solutions 
    simultaneously for multi-layered nested structures.
    
    This function handles heterogeneous list depths and preserves or transforms 
    special markers (999) based on the target layer's requirements.
    """

    # Case A: Handling Nested Structures (Recursive Depth)
    if isinstance(x, list):
        new_v_list, new_x_list, new_d_list = [], [], []
        for i in range(len(x)):
            # Recursive call to traverse deeper into the nested hierarchy
            res_v, res_x, res_d = pso_recursive_sync_update(
                v[i], x[i], pbest[i], gbest[i], w, c1, c2 ,keep_999
            )
            new_v_list.append(res_v)
            new_x_list.append(res_x)
            new_d_list.append(res_d)
        return new_v_list, new_x_list, new_d_list

    # Case B: Handling Special Identifier 999 (Inertial or Fixed Points)
    if x == 999:
        # If keep_999 is True, preserve the marker for structural logic (e.g., Layer 3 skip-nodes)
        # If False, convert to 0 to satisfy bit-sequence requirements (e.g., Layer 4 sequencing)
        if keep_999:
            return 999, 999, 999
        else:
            return 999, 999, 0

    # Case C: Reached Leaf Node (Numerical values), Execute PSO Kinematics
    r1, r2 = np.random.rand(), np.random.rand()
    
    # 1. Velocity Update with Clamping within [-6, 6]
    # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    new_v = (w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x))
    new_v = max(min(new_v, 6), -6)
    
    # 2. Continuous Position Update
    # x_new = x + v_new
    new_x = x + new_v
    
    # 3. Discrete Mapping based on Sigmoid Threshold of 0.5
    # Mathematically equivalent to checking if the continuous position is positive
    new_d = 1 if new_x > 0 else 0
    
    return new_v, new_x, new_d