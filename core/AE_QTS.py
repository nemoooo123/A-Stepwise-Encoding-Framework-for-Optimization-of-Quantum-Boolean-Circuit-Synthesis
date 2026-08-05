import math
import numpy as np
import matplotlib.pyplot as plt
from utils.init_state import gen_nbrs
from utils.topology import decode_and_synthesize,verify_circuit_logic


def _shannon_entropy_of_distribution(prob_dist):
    """
    Compute the Shannon entropy (base-2) of a single probability distribution.
    H = -sum(p * log2(p)) for p > 0.
    """
    entropy = 0.0
    for p in prob_dist:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def mean_shannon_entropy(quantum_state):
    """
    Traverse a hierarchical quantum state (nested probability layers) and return
    the average Shannon entropy over every valid probability distribution.
    [999, 999] sentinel entries are skipped.
    """
    entropies = []

    def _collect(node):
        if isinstance(node, np.ndarray):
            if node.ndim == 1:
                if not np.any(node == 999):
                    entropies.append(_shannon_entropy_of_distribution(node))
            else:
                for row in node:
                    _collect(row)
        elif isinstance(node, (list, tuple)):
            # A leaf distribution is a flat list of scalar probabilities.
            if len(node) > 0 and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in node
            ):
                if 999 not in node:
                    entropies.append(_shannon_entropy_of_distribution(node))
            else:
                for child in node:
                    _collect(child)

    _collect(quantum_state)
    return sum(entropies) / len(entropies) if entropies else 0.0


def collect_entropy_blocks(quantum_state):
    """
    Flatten a hierarchical quantum state into the list of 2-D probability blocks
    whose rows are the distributions mean_shannon_entropy() would average over.

    The nesting layout is fixed for the whole experiment and updateQ only writes
    individual elements in place, so the blocks stay live views of the current
    state: collect once, then re-read them every iteration.

    Returns None if a distribution is held in a plain Python list rather than an
    ndarray (which cannot be tracked as a view), signalling the caller to fall
    back to mean_shannon_entropy().
    """
    blocks = []

    def _collect(node):
        if isinstance(node, np.ndarray):
            if node.ndim == 1:
                if not np.any(node == 999):
                    blocks.append(node.reshape(1, -1))
            elif np.any(node == 999):
                # Mixed block: keep only the sentinel-free rows, as _collect does.
                for row in node:
                    _collect(row)
            else:
                blocks.append(node)
        elif isinstance(node, (list, tuple)):
            # A leaf distribution is a flat list of scalar probabilities.
            if len(node) > 0 and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in node
            ):
                if 999 not in node:
                    # Not an ndarray, so it cannot be tracked as a live view.
                    raise _UntrackableState()
            else:
                for child in node:
                    _collect(child)

    try:
        _collect(quantum_state)
    except _UntrackableState:
        return None
    return blocks


class _UntrackableState(Exception):
    """Raised internally when a quantum state cannot be viewed as ndarray blocks."""


def mean_shannon_entropy_cached(quantum_state, blocks):
    """
    Vectorized equivalent of mean_shannon_entropy() using the blocks previously
    collected by collect_entropy_blocks(). Falls back to the traversal-based
    implementation when no blocks were collected.
    """
    if blocks is None:
        return mean_shannon_entropy(quantum_state)
    if not blocks:
        return 0.0

    probs = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
    logs = np.zeros_like(probs, dtype=float)
    np.log2(probs, out=logs, where=probs > 0)
    return float(-(probs * logs).sum() / probs.shape[0])


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
                                 unique_history_matrix,
                                 a1_history_matrix,
                                 a2_history_matrix,
                                 entropy1_history_matrix,
                                 entropy2_history_matrix,
                                 entropy3_history_matrix,
                                 entropy4_history_matrix,
                                 mode_count_history_matrix,
                                 target_output,
                                 delta_theta):
    """
    Executes a single trial of the AE-QTS algorithm.
    Iteratively updates quantum individuals (qindividuals1-4) to minimize circuit gate count.
    """
    num_cycles = len(rotation_cycles)
    current_iter = 0
    global_best_gate_count = float('inf')
    global_best_unique_count = float('inf')
    global_best_a1_count = float('inf')
    global_best_a2_count = float('inf')
    global_best_circuit = []

    # Collect the entropy measurement views once: the quantum states keep their
    # nesting layout for the whole run and are only ever updated element-wise,
    # so re-walking them every iteration was pure overhead.
    entropy_blocks = [
        collect_entropy_blocks(q)
        for q in (qindividuals1, qindividuals2, qindividuals3, qindividuals4)
    ]

    while current_iter < max_iterations:
        current_iter += 1
        
        # Step 1: Neighborhood Generation
        # Create candidate solutions (neighbors) based on the current quantum state of qindividuals
        nbr1, nbr2, nbr3, nbr4 = gen_nbrs(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, num_neighbors
        )

        # Step 2: Decoding and Circuit Synthesis
        # Convert quantum neighbors into concrete reversible circuit solutions
        circuit_solutions = decode_and_synthesize(
            nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory,current_iter
        )
        # Step 3: Fitness Evaluation (Gate Count Analysis)
        # Pair each solution's gate count with its original neighborhood index
        # Lower scores indicate superior individuals in our minimization framework.
        solution_metrics = [(sol_tuple[2], sol_tuple[1],sol_tuple[3],sol_tuple[4],sol_tuple[5], idx) for idx, sol_tuple in enumerate(circuit_solutions)]
        
        # Sort by unique gate count first, then total gate count as tiebreaker, both ascending
        # sorted_metrics = sorted(solution_metrics, key=lambda x: (x[1],x[0]))
        #原始版本
        sorted_metrics = sorted(solution_metrics, key=lambda x: x[1])
        sorted_metrics_getbest = sorted(solution_metrics, key=lambda x: x[1])

        # Diversity/convergence measure: among this iteration's neighbor solutions,
        # how many share the single most-frequent score (total gate count)?
        # A rising mode count (toward num_neighbors) means the population is
        # collapsing onto one score, i.e. the search is converging.
        iter_scores = [m[1] for m in solution_metrics]
        _, _mode_counts = np.unique(iter_scores, return_counts=True)
        mode_count_history_matrix[experiment_id][current_iter - 1] = int(_mode_counts.max())
        #-----
        local_best_idx = sorted_metrics_getbest[0][5]
        local_best_gate_count = circuit_solutions[local_best_idx][1]
        local_best_unique_count = circuit_solutions[local_best_idx][2]
        local_best_a1_count = circuit_solutions[local_best_idx][3]
        local_best_a2_count = circuit_solutions[local_best_idx][4]
        local_best_circuit = circuit_solutions[local_best_idx][0]
        #-----
        # local_best_idx = sorted_metrics[0][4]
        # local_best_gate_count = circuit_solutions[local_best_idx][1]
        # local_best_unique_count = circuit_solutions[local_best_idx][2]
        # local_best_a1_count = circuit_solutions[local_best_idx][3]
        # local_best_a2_count = circuit_solutions[local_best_idx][4]
        # local_best_circuit = circuit_solutions[local_best_idx][0]
        # Step 4: Quantum Population Update (Angle Adjustment)
        # Update the probability amplitudes of qindividuals based on neighbor performance
        updateQ(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
            num_neighbors, nbr1, nbr2, nbr3, nbr4, 
            [m[5] for m in sorted_metrics], num_cycles, delta_theta
        )

        # Step 4.5: Quantum State Convergence Measurement
        # Record the mean Shannon entropy of each updated quantum state for this
        # iteration. Lower entropy => that state is more converged.
        entropy1_history_matrix[experiment_id][current_iter - 1] = mean_shannon_entropy_cached(qindividuals1, entropy_blocks[0])
        entropy2_history_matrix[experiment_id][current_iter - 1] = mean_shannon_entropy_cached(qindividuals2, entropy_blocks[1])
        entropy3_history_matrix[experiment_id][current_iter - 1] = mean_shannon_entropy_cached(qindividuals3, entropy_blocks[2])
        entropy4_history_matrix[experiment_id][current_iter - 1] = mean_shannon_entropy_cached(qindividuals4, entropy_blocks[3])

        # Step 5: Global Best Tracking
        # Update the overall best solution if a new minimum gate count is discovered
        if global_best_gate_count > local_best_gate_count:
            global_best_gate_count = local_best_gate_count
            global_best_unique_count = local_best_unique_count
            global_best_a1_count = local_best_a1_count
            global_best_a2_count = local_best_a2_count
            global_best_circuit = local_best_circuit
            
        # Step 6: Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol[0], num_bits, target_output) for sol in circuit_solutions)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # Step 7: Data Recording for Statistical Analysis
        # Record the current best gate count into the fitness history matrix (used for np.mean later)
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count
        unique_history_matrix[experiment_id][current_iter - 1] = global_best_unique_count
        a1_history_matrix[experiment_id][current_iter - 1] = global_best_a1_count
        a2_history_matrix[experiment_id][current_iter - 1] = global_best_a2_count


    return fitness_history_matrix, unique_history_matrix, a1_history_matrix, a2_history_matrix, entropy1_history_matrix, entropy2_history_matrix, entropy3_history_matrix, entropy4_history_matrix, mode_count_history_matrix, global_best_gate_count, global_best_circuit

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
                    qindividuals1[i][j][b_val] += delta_theta/(t+1)
                    qindividuals1[i][j][w_val] -= delta_theta/(t+1)
                    
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
                    qindividuals2[i][j][best_sol2[i][j]] += delta_theta/(t+1)
                    qindividuals2[i][j][worst_sol2[i][j]] -= delta_theta/(t+1)
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
                                qindividuals3[i][j][k][l][b_bit] += (delta_theta/(t+1))
                                qindividuals3[i][j][k][l][w_bit] -= (delta_theta/(t+1))
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
                            qindividuals4[i][j][k][b_gate] += delta_theta/(t+1)
                            qindividuals4[i][j][k][w_gate] -= delta_theta/(t+1)
                            if qindividuals4[i][j][k][w_gate] <= 0:
                                qindividuals4[i][j][k][b_gate] = 1.0
                                qindividuals4[i][j][k][w_gate] = 0.0
        t += 1

          
   