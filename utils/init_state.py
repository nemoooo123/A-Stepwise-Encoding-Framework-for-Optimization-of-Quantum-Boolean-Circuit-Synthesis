import numpy as np
import random
import copy
from math import ceil, log2

def find_cycles(permutation, check_zero_gate=True):
    """
    Perform cycle decomposition on a given permutation.
    Identifies disjoint cycles to represent the reversible function.
    """
    # Track whether each element has been visited to avoid redundant processing
    visited = [False] * len(permutation) 
    cycles = []
    
    for i in range(len(permutation)):
        if not visited[i]:
            # Start a new cycle from the unvisited element
            current = i
            cycle = []

            while not visited[current]:
                # Traverse the permutation path until the cycle closes
                visited[current] = True
                cycle.append(current)
                # Jump to the next element based on the permutation mapping
                current = permutation[current] 
            
            # Filter out identity mappings (cycles of length 1)
            if len(cycle) > 1:
                cycles.append(cycle)

    # If no cycles exist (beyond length 1), it is an identity mapping
    if len(cycles) == 0:
        check_zero_gate = False

    return cycles, check_zero_gate

def hamming_distance(a, b):
    """
    Calculate the Hamming distance between two integers.
    Measures the number of bit positions at which the corresponding bits are different.
    """
    return bin(a ^ b).count('1') 

def build_encode(cycles):
    """
    Initialize multi-level probability matrices for the optimization framework.
    Constructs a four-layer encoding structure to represent cycle selection,
    edge decomposition, path transformation, and gate sequencing.
    """
    cycles_bit = len(cycles)
    
    # Calculate required encoding bits for cycle selection
    if cycles_bit == 1:
        bits = 1
    else:
        bits = ceil(log2(cycles_bit))

    # Layer 1: Selection probability for different cycles
    prob_layer_cycle_select = [np.full((bits, 2), 0.5) for _ in range(cycles_bit)]
    
    # Encoding for edge selection within a cycle and gate ordering
    # Layer 2: Probability matrix for identifying the "breaking edge" (starting point) 
    # within each cycle. This determines the initial node of the decomposition.
    prob_layer_edge_break=[]
    # Layer 3: Intermediate node selection. When the Hamming distance between two nodes
    # in a swap exceeds 1 bit, this layer encodes the specific path/intermediate nodes
    # within the Boolean hypercube.
    prob_layer_path_nodes=[]
    prob_layer_seq_order=[] #Layer 4: Sequence directionality

    # node_mapping_index: Stores the actual number of nodes in each cycle.
    # Since the measurement (sampling) uses a binary representation (powers of 2),
    # the binary space is often larger than the actual cycle length. This table 
    # acts as a mapping constraint to ensure sampled results correctly align 
    # with valid cycle nodes and prevent indexing errors during edge breaking.
    node_mapping_index=[] 
    # transition_trajectories: Stores the "transition_trajectoriesition Path" for each cycle.
    # It records the sequence of nodes after cycle reversal and 
    # closure (returning to the start node), providing the exact 
    # trajectory for reversible gate synthesis.
    transition_trajectories=[]
    for i in cycles:
        #Determine bit requirements for internal cycle edges
        cycle_length=len(i)
        if cycle_length==2:
            node_mapping_index.append(1)
        else:node_mapping_index.append(cycle_length)
        
        required_node_bits=ceil(log2(cycle_length))
        

        # Calculate encoding bits required for nodes within the current cycle
        cycle_edge_prob_matrix=np.zeros((required_node_bits,2))
        cycle_edge_prob_matrix.fill(1 / 2)
        prob_layer_edge_break.append(cycle_edge_prob_matrix)
        
        # Calculate Hamming distances between elements to generate Layer 3
        cycle_path_probs=[] #Nested list for a single cycle's Layer 3 data
        cycle_seq_probs=[] # Nested list for a single cycle's Layer 4 data

        i.reverse() # Reverse the cycle to ensure correct circuit synthesis order
        
        length=len(i)
        ic=copy.copy(i)
        i.append(i[0])# Close the cycle
        transition_trajectories.append(i)
        # If the cycle is not a simple swap (length != 2)
        if length!=2:
            for x in range(length):
                start=i[x]
                end=i[x+1]  
                # Identify Hamming distance between consecutive states
                dist_hamming=hamming_distance(start,end)
                required_encoding_bits = ceil(log2(dist_hamming)) 
                
                # print(required_encoding_bits)
                step_path_probs=[]
                if required_encoding_bits==0:
                    # Identifier for 1-bit difference (no complex sequence needed)
                    step_path_probs.append([999,999])
                    cycle_path_probs.append(step_path_probs)
                else:
                    step_path_prob_matrix=np.zeros((required_encoding_bits,2)) # Layer 3: Probability matrix for permutation selection
                    step_path_prob_matrix.fill(1 / 2)
                    for step_idx in range(dist_hamming):
                        step_path_probs.append(step_path_prob_matrix)
                    cycle_path_probs.append(step_path_probs)

                # Layer 4: Sequence order probability (Head vs Tail movement)
                # Formula derived for gate count based on bit difference
                bit_gate_seq = 2 * dist_hamming - 1 - 1 

                if bit_gate_seq==0: 
                    step_seq_prob_matrix=[]
                    step_seq_prob_matrix.append([999,999]) 
                    cycle_seq_probs.append(step_seq_prob_matrix)
                else:
                    step_seq_prob_matrix=np.zeros((bit_gate_seq,2))
                    step_seq_prob_matrix.fill(1 / 2)
                    cycle_seq_probs.append(step_seq_prob_matrix)

                
            
        else:
            # Handle simple 2-element swap case
            start=i[0]
            end=i[1]
            dist_hamming=hamming_distance(start,end)
            required_encoding_bits = ceil(log2(dist_hamming))

            step_path_probs=[]
            if required_encoding_bits==0:
                step_path_probs.append([999,999])
                cycle_path_probs.append(step_path_probs)
            else:
                step_path_prob_matrix=np.zeros((required_encoding_bits,2)) 
                step_path_prob_matrix.fill(1 / 2)
                for q33 in range(dist_hamming):
                    step_path_probs.append(step_path_prob_matrix)
                cycle_path_probs.append(step_path_probs)

            bit_gate_seq = 2 * dist_hamming - 1 - 1 

            if bit_gate_seq==0: 
                step_seq_prob_matrix=[]
                step_seq_prob_matrix.append([999,999]) 
                cycle_seq_probs.append(step_seq_prob_matrix)
            else:
                step_seq_prob_matrix=np.zeros((bit_gate_seq,2))
                step_seq_prob_matrix.fill(1 / 2)
                cycle_seq_probs.append(step_seq_prob_matrix)
            

        prob_layer_path_nodes.append(cycle_path_probs)
        prob_layer_seq_order.append(cycle_seq_probs) 

    return prob_layer_cycle_select, prob_layer_edge_break, prob_layer_path_nodes, prob_layer_seq_order, node_mapping_index, transition_trajectories

def gen_nbrs(prob_layer_L1, prob_layer_L2, prob_layer_L3, prob_layer_L4, population_size):
    """
    Generate candidate solutions by sampling from the hierarchical probability layers.
    This process creates a population of 'neighbors' or 'candidate paths' in the 
    combinatorial search space.
    """
    # Sample Layer 1: Global selection candidates
    candidates_L1 = [np.array(sample_layer_L1(prob_layer_L1)) for _ in range(population_size)]
    
    # Sample Layer 2: Component entry point (break edge) candidates
    candidates_L2 = [sample_layer_L2(prob_layer_L2) for _ in range(population_size)]  
    
    # Sample Layer 3: Intermediate path node candidates
    candidates_L3 = [sample_layer_L3(prob_layer_L3) for _ in range(population_size)]
    
    # Sample Layer 4: Gate sequencing/order candidates
    candidates_L4 = [sample_layer_L4(prob_layer_L4) for _ in range(population_size)]
    
    return candidates_L1, candidates_L2, candidates_L3, candidates_L4

def sample_layer_L1(prob_matrices): 
    """
    Perform probabilistic sampling on Layer 1 matrices to generate a discrete solution.
    Each matrix represents the selection probability for a specific cycle component.
    """
    # selection_results: Stores the sampled binary strings for all cycles
    selection_results=[] 
    # Iterate through each cycle's probability matrix
    for cycle_idx in range(len(prob_matrices)): 
        binary_code = []
        
        # Iterate through each bit in the current cycle's probability matrix
        for bit_idx in range(len(prob_matrices[cycle_idx])): 
            # Generate a random float between 0 and 1 for stochastic sampling
            random_threshold = np.random.rand(1)
            
            # Compare the stored probability against the random threshold
            # If the probability of state 0 is less than the threshold, select state 1
            if prob_matrices[cycle_idx][bit_idx][0] < random_threshold:
                binary_code.append(1)
            else:
                binary_code.append(0)
        
        selection_results.append(binary_code)
    
    
    
    return selection_results

def sample_layer_L2(prob_matrices_L2):
    """
    Sample from Layer 2 probability matrices to determine the 'break edge'
    (starting point) for each cycle component.
    Returns a list of binary codes representing the chosen entry nodes.
    """
    # break_edge_codes: Stores the sampled binary strings for identifying entry nodes
    break_edge_codes = []
    
    # Iterate through each cycle's probability matrix in Layer 2
    for cycle_idx in range(len(prob_matrices_L2)):
        # binary_string: The sampled bits for the current cycle's entry point
        binary_string = []
        
        # Iterate through each bit's probability distribution 
        for bit_prob in prob_matrices_L2[cycle_idx]:
            # Generate a stochastic threshold for binary sampling
            random_threshold = np.random.rand(1)
            
            # Binary selection based on the probability of state 0
            if bit_prob[0] < random_threshold:
                binary_string.append(1)
            else:
                binary_string.append(0)

        break_edge_codes.append(binary_string)
    
    return break_edge_codes

def sample_layer_L3(prob_matrices_L3):
    """
    Sample from Layer 3 probability matrices to determine intermediate path nodes.
    This handles multi-bit Hamming distance transitions by selecting specific nodes
    within the Boolean hypercube for each transformation step.
    """
    # path_selection_results: Stores sampled nodes for all cycles and their steps
    path_selection_results = []
    
    # Iterate through each cycle component
    for cycle_idx in range(len(prob_matrices_L3)):
        cycle_path_samples = []
        
        # Iterate through each transition step (swap/transformation) within the cycle
        for step_idx in range(len(prob_matrices_L3[cycle_idx])):
            step_node_codes = []
            
            # Identify if the current step is a direct transition (indicated by [999,999])
            # num_substeps represents the number of intermediate points in this transition
            num_substeps = len(prob_matrices_L3[cycle_idx][step_idx])
            
            for substep_matrix in prob_matrices_L3[cycle_idx][step_idx]:
                # binary_node: The sampled binary string for an intermediate state
                binary_node = []
                
                if num_substeps == 1:
                    # Identifier 999 indicates a direct 1-bit Hamming distance (no intermediate node needed)
                    step_node_codes.append(999)
                else:
                    # Perform bit-wise stochastic sampling for the intermediate node
                    for bit_prob in substep_matrix:
                        random_threshold = np.random.rand(1)
                        
                        if bit_prob[0] < random_threshold:
                            binary_node.append(1)
                        else:
                            binary_node.append(0)
                    
                    # Store the sampled binary representation for this specific node
                    step_node_codes.append(binary_node)
            
            # Append the collection of nodes for this step to the cycle's path
            cycle_path_samples.append(step_node_codes)
            
        # Append the complete cycle path to the global result list
        path_selection_results.append(cycle_path_samples)

    return path_selection_results

def sample_layer_L4(prob_matrices_L4):
    """
    Sample from Layer 4 probability matrices to determine the operational sequence.
    Each sample represents the directionality and ordering of transformation gates,
    followed by a repair mechanism to ensure sequence validity.
    """
    # sequence_selection_results: Stores the final ordered sequences for all cycles
    sequence_selection_results = []
    
    # Iterate through each cycle component
    for cycle_idx in range(len(prob_matrices_L4)):
        cycle_sequence_samples = []
        
        # Iterate through each transition step within the cycle
        for step_idx in range(len(prob_matrices_L4[cycle_idx])):
            step_direction_bits = []
            
            # Iterate through the probability distribution for each sequencing decision
            for bit_idx in range(len(prob_matrices_L4[cycle_idx][step_idx])):
                # Generate a random threshold for stochastic sampling
                random_threshold = np.random.rand(1)
                
                # prob_value: The specific probability for this sequencing bit
                prob_value = prob_matrices_L4[cycle_idx][step_idx][bit_idx][0]
                
                if prob_value == 999:
                    # Identifier 999 indicates a default/fixed sequence (minimal Hamming distance)
                    step_direction_bits.append(0)
                elif prob_value > random_threshold:
                    # Probability condition met: Assign first directionality state
                    step_direction_bits.append(0)
                else:
                    # Probability condition not met: Assign second directionality state
                    step_direction_bits.append(1)
            
            # Constraint Satisfaction: Repair the sequence bits to maintain logical consistency
            # (e.g., ensuring balanced transformation gates)
            repaired_sequence = repair_sequence_logic(step_direction_bits)
            
            cycle_sequence_samples.append(repaired_sequence)
        
        # Collect the validated sequences for this cycle
        sequence_selection_results.append(cycle_sequence_samples)
    
    return sequence_selection_results

def repair_sequence_logic(sequence_bits):
    """
    Repair mechanism to ensure balance between binary states in Layer 4.
    Adjusts the number of 0s and 1s to maintain logical consistency for 
    reversible transformation sequences.
    """
    # If the sequence length is 1, no repair is necessary (fixed logic)
    if len(sequence_bits) <= 1:
        return sequence_bits

    count_zero = sequence_bits.count(0)
    count_one = sequence_bits.count(1)
    
    # Calculate the discrepancy between states
    # imbalance > 0 means too many zeros; imbalance < 0 means too many ones
    imbalance = int((count_zero - count_one) / 2)

    if imbalance > 0:
        # Case: Excessive zeros. Randomly convert 'imbalance' zeros to ones.
        num_to_flip = imbalance
        # Identify indices of all zero states
        zero_indices = [i for i, bit in enumerate(sequence_bits) if bit == 0]
        
        # Stochastic selection of indices for state flipping
        flip_indices = random.sample(zero_indices, k=num_to_flip)
        for idx in flip_indices:
            sequence_bits[idx] = 1
            
    elif imbalance < 0:
        # Case: Excessive ones. Randomly convert 'num_to_flip' ones to zeros.
        num_to_flip = int((count_one - count_zero) / 2)
        # Identify indices of all one states
        one_indices = [i for i, bit in enumerate(sequence_bits) if bit == 1]
        
        # Stochastic selection of indices for state flipping
        flip_indices = random.sample(one_indices, k=num_to_flip)
        for idx in flip_indices:
            sequence_bits[idx] = 0
            
    return sequence_bits


def gen_pso_init_states(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, population_size):
    """
    專門為 PSO 設計的初始化函式。
    回傳與位置結構相同的連續隨機速度矩陣 (範圍 [-6, 6])。
    """
    # 粒子群位置容器 (Continuous values for optimization)
    pso_positions_l1, pso_positions_l2, pso_positions_l3, pso_positions_l4 = [], [], [], []
    
    # 粒子群速度容器 (Movement vectors)
    pso_velocities_l1, pso_velocities_l2, pso_velocities_l3, pso_velocities_l4 = [], [], [], []
    
    # 離散化後的解容器 (Mapped 0/1 bits for circuit synthesis)
    discrete_solutions_l1, discrete_solutions_l2, discrete_solutions_l3, discrete_solutions_l4 = [], [], [], []
    
    for _ in range(population_size):
        # Layer 1: 全域選擇層
        pos1, v1, disc1 = pso_layer_L1(pop_matrix1)
        pso_positions_l1.append(pos1)
        pso_velocities_l1.append(v1)
        discrete_solutions_l1.append(disc1)

        # Layer 2: 進入點選擇層
        pos2, v2, disc2 = pso_layer_L2(pop_matrix2)
        pso_positions_l2.append(pos2)
        pso_velocities_l2.append(v2)
        discrete_solutions_l2.append(disc2)

        # Layer 3: 中間路徑節點層
        pos3, v3, disc3 = pso_layer_L3(pop_matrix3)
        pso_positions_l3.append(pos3)
        pso_velocities_l3.append(v3)
        discrete_solutions_l3.append(disc3)

        # Layer 4: 門序列順序層
        pos4, v4, disc4 = pso_layer_L4(pop_matrix4)
        pso_positions_l4.append(pos4)
        pso_velocities_l4.append(v4)
        discrete_solutions_l4.append(disc4)
        
    # 以元組形式回傳所有層級的數據
    return (pso_positions_l1, pso_velocities_l1, discrete_solutions_l1,
            pso_positions_l2, pso_velocities_l2, discrete_solutions_l2,
            pso_positions_l3, pso_velocities_l3, discrete_solutions_l3,
            pso_positions_l4, pso_velocities_l4, discrete_solutions_l4)

def pso_layer_L1(population_templates): 
    """
    Initialize Particle Swarm Optimization (PSO) Layer 1 components.
    Generates:
    1. Continuous initial positions within [-6, 6].
    2. Initial velocity vectors within [-1, 1].
    3. Discrete binary solutions based on a 0.5 sigmoid threshold (x > 0).
    """
    # initial_positions: Latent coordinates for the swarm to "fly" through (Continuous)
    initial_positions = [] 
    # initial_velocities: Movement vectors for the particles
    initial_velocities = []
    # initial_discrete_solutions: The actual 0/1 bits used for circuit synthesis (Discrete)
    initial_discrete_solutions = []

    # Iterate through each cycle's encoding structure
    for cycle_idx in range(len(population_templates)): 
        pos_code = []
        vel_code = []
        discrete_code = []
        
        # Iterate through each bit position defined in the template
        for bit_idx in range(len(population_templates[cycle_idx])): 
            # 1. Initialize Continuous Position: [-6, 6]
            stochastic_pos = np.random.uniform(-6, 6)
            
            # 2. Initialize Velocity: [-1, 1]
            stochastic_vel = np.random.uniform(-1, 1)
            
            # 3. Generate Discrete Solution: 
            # Based on your requirement: If sigmoid(x) > 0.5 (which means x > 0), set to 1.
            discrete_bit = 1 if stochastic_pos > 0 else 0
            
            pos_code.append(stochastic_pos)
            vel_code.append(stochastic_vel)
            discrete_code.append(discrete_bit)
        
        initial_positions.append(pos_code)
        initial_velocities.append(vel_code)
        initial_discrete_solutions.append(discrete_code)
    
    # Return all three components for the PSO population initialization
    return initial_positions, initial_velocities, initial_discrete_solutions

def pso_layer_L2(entry_point_templates):
    """
    Initialize Particle Swarm Optimization (PSO) Layer 2 components.
    Generates:
    1. Continuous initial positions within [-6, 6] for entry points.
    2. Initial velocity vectors within [-1, 1].
    3. Discrete entry point bits based on a 0.5 sigmoid threshold (x > 0).
    """
    # initial_positions: Latent coordinates for entry point optimization
    initial_positions = []
    # initial_velocities: Movement vectors for the swarm to explore the entry node space
    initial_velocities = []
    # initial_discrete_solutions: The binary entry point bits (0/1) for circuit synthesis
    initial_discrete_solutions = []
    
    # Iterate through each cycle's entry point structure in Layer 2
    for cycle_idx in range(len(entry_point_templates)):
        pos_code = []
        vel_code = []
        discrete_code = []
        
        # Iterate through each bit defined in the entry point template 
        for _ in entry_point_templates[cycle_idx]:
            # 1. Initialize Continuous Position: Sigmoid sensitivity range [-6, 6]
            stochastic_pos = np.random.uniform(-6, 6)
            
            # 2. Initialize Velocity: Initial search momentum [-1, 1]
            stochastic_vel = np.random.uniform(-1, 1)
            
            # 3. Generate Discrete Solution: 
            # Sigmoid(x) > 0.5 is mathematically equivalent to x > 0
            discrete_bit = 1 if stochastic_pos > 0 else 0
            
            pos_code.append(stochastic_pos)
            vel_code.append(stochastic_vel)
            discrete_code.append(discrete_bit)

        initial_positions.append(pos_code)
        initial_velocities.append(vel_code)
        initial_discrete_solutions.append(discrete_code)
    
    # Return synchronized Position, Velocity, and Discrete Solution for Layer 2 Swarm
    return initial_positions, initial_velocities, initial_discrete_solutions

def pso_layer_L3(path_node_templates):
    """
    Initialize Particle Swarm Optimization (PSO) Layer 3 components.
    Generates:
    1. Continuous positions within [-6, 6] for path nodes.
    2. Initial velocity vectors within [-1, 1].
    3. Discrete node bits (0/1) based on a 0.5 sigmoid threshold (x > 0).
    Special markers (999) are preserved across all structures.
    """
    # initial_positions: Latent coordinates for the swarm to explore path combinations
    initial_positions = []
    # initial_velocities: Trajectory vectors for the path optimization
    initial_velocities = []
    # initial_discrete_solutions: Binary node representations for circuit logic evaluation
    initial_discrete_solutions = []

    # Iterate through each cycle component in Layer 3
    for cycle_idx in range(len(path_node_templates)):
        cycle_pos, cycle_vel, cycle_disc = [], [], []
        
        # Iterate through each transition step within the cycle
        for step_idx in range(len(path_node_templates[cycle_idx])):
            step_pos, step_vel, step_disc = [], [], []
            
            # Determine if this step is a direct transition (999) or requires intermediate nodes
            substep_count = len(path_node_templates[cycle_idx][step_idx])
            
            for node_template in path_node_templates[cycle_idx][step_idx]:
                if substep_count == 1:
                    # Identifier 999: Direct Hamming distance jump. 
                    # Preserve 999 in all three outputs to maintain structural alignment.
                    step_pos.append(999)
                    step_vel.append(999)
                    step_disc.append(999)
                else:
                    node_pos, node_vel, node_disc = [], [], []
                    
                    # Perform bit-wise initialization for the node
                    for _ in node_template:
                        # 1. Continuous Position: [-6, 6]
                        stochastic_pos = np.random.uniform(-6, 6)
                        # 2. Velocity: [-1, 1]
                        stochastic_vel = np.random.uniform(-1, 1)
                        # 3. Discrete Solution: x > 0 is equivalent to Sigmoid(x) > 0.5
                        discrete_bit = 1 if stochastic_pos > 0 else 0
                        
                        node_pos.append(stochastic_pos)
                        node_vel.append(stochastic_vel)
                        node_disc.append(discrete_bit)
                    
                    step_pos.append(node_pos)
                    step_vel.append(node_vel)
                    step_disc.append(node_disc)
            
            cycle_pos.append(step_pos)
            cycle_vel.append(step_vel)
            cycle_disc.append(step_disc)
            
        initial_positions.append(cycle_pos)
        initial_velocities.append(cycle_vel)
        initial_discrete_solutions.append(cycle_disc)

    # Return synchronized Continuous Position, Velocity, and Discrete Solution for Layer 3
    return initial_positions, initial_velocities, initial_discrete_solutions

def pso_layer_L4(sequence_templates):
    """
    Initialize Particle Swarm Optimization (PSO) Layer 4 components.
    Generates:
    1. Continuous initial positions within [-6, 6] for sequencing decisions.
    2. Initial velocity vectors within [-1, 1].
    3. Repaired discrete binary solutions (0/1) for valid gate ordering.
    """
    # initial_positions: Latent coordinates for optimizing gate sequences
    initial_positions = []
    # initial_velocities: Trajectory vectors for the sequencing optimization
    initial_velocities = []
    # initial_discrete_solutions: Validated (repaired) binary sequences for circuit synthesis
    initial_discrete_solutions = []
    
    # Iterate through each cycle's sequencing structure in Layer 4
    for cycle_idx in range(len(sequence_templates)):
        cycle_pos, cycle_vel, cycle_disc = [], [], []
        
        # Iterate through each transition step within the cycle
        for step_idx in range(len(sequence_templates[cycle_idx])):
            step_pos, step_vel, step_raw_disc = [], [], []
            
            # Iterate through the sequencing bits defined in the template
            for bit_template in sequence_templates[cycle_idx][step_idx]:
                # Check for the 999 fixed-sequence identifier
                if bit_template[0] == 999:
                    stochastic_pos = 999
                    stochastic_vel = 999
                    discrete_bit = 0 # Default state for fixed sequences (per original logic)
                else:
                    # 1. Initialize Position: [-6, 6]
                    stochastic_pos = np.random.uniform(-6, 6)
                    # 2. Initialize Velocity: [-1, 1]
                    stochastic_vel = np.random.uniform(-1, 1)
                    # 3. Initial Discrete Bit: Based on 0.5 threshold (x > 0)
                    discrete_bit = 1 if stochastic_pos > 0 else 0
                
                step_pos.append(stochastic_pos)
                step_vel.append(stochastic_vel)
                step_raw_disc.append(discrete_bit)
            
            # --- 重要：執行序列修復邏輯 ---
            # 確保初始化的二進位序列符合電路邏輯（例如閘的正反向平衡）
            repaired_disc = repair_sequence_logic(step_raw_disc)
            
            cycle_pos.append(step_pos)
            cycle_vel.append(step_vel)
            cycle_disc.append(repaired_disc)
            
        initial_positions.append(cycle_pos)
        initial_velocities.append(cycle_vel)
        initial_discrete_solutions.append(cycle_disc)
    
    # Return synchronized Position, Velocity, and Validated Discrete Solution for Layer 4
    return initial_positions, initial_velocities, initial_discrete_solutions