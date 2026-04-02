import numpy as np
import random
import copy
import time
from utils.init_state import hamming_distance


def decode_and_synthesize(pop_l1, pop_l2, pop_l3, pop_l4, mapping_table, num_units, pop_size, trajectories):
    """
    Decodes hierarchical binary samples into decimal values and synthesizes 
    quantum circuit routes for the entire population.
    
    Args:
        pop_l1-l4: Binary samples for 4 layers.
        mapping_table: Constraint table for valid cycle lengths (encode_2_bit_table).
        num_qubits: Number of qubits (n).
        pop_size: Total individuals in population (N).
        trajectories: Database of transition paths (trans).
    """
    circuit_solutions = []
    
    for i in range(pop_size):
        # Layer 1: Component priority decoding (Binary to Decimal)
        decoded_l1 = [int(''.join(map(str, bits)), 2) for bits in pop_l1[i]]
            
        # Layer 2: Entry point decoding with modulo constraint handling
        decoded_l2 = []
        for idx, bits in enumerate(pop_l2[i]):
            val = int(''.join(map(str, bits)), 2)
            limit = mapping_table[idx]
            # Ensure the starting index is within the valid range of the cycle
            decoded_l2.append(val % limit if val >= limit else val)

        # Layer 3: Intermediate node decoding (Handling special 999 bypass flag)
        decoded_l3 = []
        for cycle in pop_l3[i]:
            cycle_steps = []
            for step in cycle:
                if step[0] == 999:
                    cycle_steps.append([999])
                else:
                    # Convert each node bit-string to a decimal index
                    cycle_steps.append([int(''.join(map(str, node)), 2) for node in step])
            decoded_l3.append(cycle_steps)
        
        # Synthesis: Transform decoded parameters into the final circuit structure
        # Passing Layer 4 directly as it is handled within the synthesize_route logic
        
        individual_solution = synthesize_route(decoded_l1, decoded_l2, decoded_l3, 
                                          pop_l4[i], num_units, trajectories)
        
        circuit_solutions.append(individual_solution)
        
    return circuit_solutions

def synthesize_route(priority_weights, entry_points, mid_node_matrix, operation_sequences, num_units, trajectories): 
    """
    Synthesizes the final execution route by analyzing cycle sequences and inter-cycle dependencies.
    General-purpose implementation for path optimization algorithms.
    
    Args:
        priority_weights (list): Layer 1 - Processing priority for each cycle.
        entry_points (list): Layer 2 - Starting index for each cycle sequence.
        mid_node_matrix (list): Layer 3 - Matrix of intermediate node indices.
        operation_sequences (list): Layer 4 - Sequence of logic operations/actions.
        num_units (int): Total number of processing units/elements (n).
        trajectories (list): Database of pre-calculated state transitions.
    """
    # Initialize Layer 3 nodes to base state
    initialize_solution_layer(mid_node_matrix)
    
    
    # Sort cycles based on Layer 1 priority (Descending)
    sorted_order = [idx for idx, _ in sorted(enumerate(priority_weights), key=lambda x: x[1], reverse=True)]

    #0201每一個解 用斷邊和循環順序，先建立位置
    final_ops=[]
    final_paths=[] #刷新裝
     
    # Core Data Structures for Path Analysis
    path_indices = []       # Trajectory index mapping
    path_weights = []        # Distance/Hamming metrics
    
    # Inter-cycle connectivity buffers
    head_states = []         # Initial state of the segment
    tail_states = []         # Final state of the segment

    # --- Phase 1: Segment Scanning & Path Construction ---
    for ind, cycle_idx in enumerate(sorted_order):
        step_indices = []
        step_weights = []
        
        start_pos = entry_points[cycle_idx]
        num_steps = len(mid_node_matrix[cycle_idx])
        if num_steps == 1: num_steps = 2 # Edge case for minimal transitions
        
        for step in range(num_steps - 1):
            curr_pos = (start_pos + step) % num_steps
            is_final = (step == num_steps - 2)
            
            # Construct atomic transition segments
            path, weight, h_node, t_node = analyze_bit_differences(curr_pos, mid_node_matrix[cycle_idx][curr_pos], 
                                                    trajectories[cycle_idx], is_final, num_units)
            
            if is_final:
                head_states.append(h_node)
                tail_states.append(t_node)
                
            step_indices.append(path)
            step_weights.append(weight)
            
        path_indices.append(step_indices)
        path_weights.append(step_weights)

    # --- Phase 2: Sequence Optimization & Index Adjustment ---      
    optimization_flags=[]
    reduction_indices=[]
    lookahead_indices=[]
    is_standard_flow=True

    for ind,cycle_idx in enumerate(sorted_order):

        # Use deepcopy to ensure data integrity and prevent unintended side effects on the original path_indices
        curr_indices = copy.deepcopy(path_indices[ind])
        curr_weights = copy.deepcopy(path_weights[ind])

        if is_standard_flow: 
            # Analyze sequences for potential redundancy reduction
            counts_tmp,elements_tmp,get_fi,he_fi=reduce_sequences_standard(curr_indices,curr_weights)
            optimization_flags.append(0)
            reduction_indices.append(-1) 

        else:
            # --- Strategy-Based Inter-Cycle Optimization ---
            # If standard flow is False, analyze specific reduction strategies (1-4)
            if strategy_type==3: 
                # STRATEGY 3: Independent cycle processing with node alignment
                curr_indices = copy.deepcopy(path_indices[ind])
                curr_weights = copy.deepcopy(path_weights[ind])
                # Baseline analysis without modifications
                counts_tmp,elements_tmp,get_fi,he_fi=reduce_sequences_standard(curr_indices,curr_weights)
                best_reduction_count=sum(counts_tmp)
                optimal_node=-1
                # Verify if the existing sequence already aligns with common transition nodes
                if len(elements_tmp[0])!=0 and elements_tmp[0][-1] in candidate_nodes :
                    pass
                else:
                    # Attempt manual alignment by iterating through candidate overlapping nodes
                    for val in candidate_nodes:
                        curr_indices = copy.deepcopy(path_indices[ind])
                        curr_weights = copy.deepcopy(path_weights[ind])
                        
                        # Perform targeted sequence analysis using the candidate node
                        kb_counts, kb_elements, kb_get_fi, kb_he_fi = reduce_sequences_targeted(curr_indices, curr_weights, val)
                        
                        # Update if the specific node alignment yields equal or better reduction
                        if sum(kb_counts) >= best_reduction_count:
                            counts_tmp, elements_tmp = kb_counts, kb_elements
                            get_fi, he_fi = kb_get_fi, kb_he_fi
                            best_reduction_count = sum(kb_counts)
                            optimal_node = val

                # Store the results of Strategy 3 search
                if optimal_node!=-1: #有東西
                    optimization_flags.append(strategy_type) #有
                    reduction_indices.append(optimal_node)
                else:
                    optimization_flags.append(0)
                    reduction_indices.append(-1)
                
            
            elif strategy_type in [1, 2]:
                # STRATEGY 1 & 2: Prefix-based cycle merging
                # Attempt to insert candidate nodes at the start of the sequence to trigger reductions

                best_reduction_count = 0
                optimal_node = -1

                for idx,val in enumerate(candidate_nodes):
                    
                    curr_indices = copy.deepcopy(path_indices[ind])
                    curr_weights = copy.deepcopy(path_weights[ind])

                    # Temporarily inject candidate node at the sequence header (index 0)
                    curr_weights.insert(0, 1)
                    curr_indices.insert(0,[val])

                    kb_counts, kb_elements, kb_get_fi, kb_he_fi = reduce_sequences_ordered(curr_indices, curr_weights)
                    
                    # Remove the temporary injected markers after analysis
                    kb_counts.pop(0)
                    kb_elements.pop(0)
                    curr_weights.pop(0)
                    curr_indices.pop(0)

                    # Selection logic: Store the first candidate or any candidate that improves the reduction count
                    if idx == 0 or sum(kb_counts) > best_reduction_count:
                        counts_tmp, elements_tmp = kb_counts, kb_elements
                        get_fi, he_fi = kb_get_fi, kb_he_fi
                        best_reduction_count = sum(kb_counts)
                        optimal_node = val

                # Validate if Strategy 1/2 successfully reduced the first step
                if counts_tmp[0] != 0: 
                    optimization_flags.append(strategy_type) 
                    reduction_indices.append(optimal_node)
                else:
                    optimization_flags.append(0)
                    reduction_indices.append(-1)

            elif is_leap_strategy==1:
                # STRATEGY 4: Leap-ahead head node verification
                curr_indices = copy.deepcopy(path_indices[ind])
                curr_weights = copy.deepcopy(path_weights[ind])
                
                counts_tmp,elements_tmp,get_fi,he_fi=reduce_sequences_standard(curr_indices,curr_weights)
                best_reduction_count=sum(counts_tmp)

                # Check if the sequence's entry node matches the leap target
                is_node_matched = (leap_target_node in curr_indices[0])
                
                # If no natural reduction exists but the node matches, force KB-style analysis

                if len(elements_tmp[0])==0 and is_node_matched:
                    curr_indices_2 = copy.deepcopy(path_indices[ind])
                    curr_weights_2 = copy.deepcopy(path_weights[ind])
                    
                    kb_counts,kb_elements_tmp,kb_get_fi,kb_he_fi=reduce_sequences_targeted(curr_indices_2,curr_weights_2,leap_target_node) #丟對了
                    
                    # Sub-case: No reduction found, or specific node mismatch
                    if sum(kb_counts)>=best_reduction_count: 
                        counts_tmp=kb_counts
                        best_reduction_count=sum(counts_tmp) 
                        elements_tmp=kb_elements_tmp
                        get_fi=kb_get_fi
                        he_fi=kb_he_fi
                elif len(elements_tmp[0])!=0 and is_node_matched:
                    if elements_tmp[0][-1]!=leap_target_node:
                        curr_indices_2 = copy.deepcopy(path_indices[ind])
                        curr_weights_2 = copy.deepcopy(path_weights[ind])
                        kb_counts,kb_elements_tmp,kb_get_fi,kb_he_fi=reduce_sequences_targeted(curr_indices_2,curr_weights_2,leap_target_node) 
                        
                        if sum(kb_counts)>=best_reduction_count: 
                            counts_tmp=kb_counts
                            best_reduction_count=sum(counts_tmp) 
                            elements_tmp=kb_elements_tmp
                            get_fi=kb_get_fi
                            he_fi=kb_he_fi
                
                optimization_flags.append(0)
                reduction_indices.append(-1)
        
        if ind<len(sorted_order)-1:  
            # Fetch the start nodes of the next cycle to predict overlap
            next_cycle_idx = sorted_order[ind+1]
            next_start_node = trajectories[next_cycle_idx][entry_points[next_cycle_idx]]
            next_second_node = trajectories[next_cycle_idx][entry_points[next_cycle_idx] + 1]
            
            # Map future trajectory indices and establish potential reduction strategies
            current_lookahead_map = map_next_transition_bits(next_start_node, next_second_node, num_units)
            lookahead_indices.append(current_lookahead_map)
            
            candidate_nodes,strategy_type,is_leap_strategy=determine_transition_strategy(
                next_start_node,next_second_node,head_states[ind],tail_states[ind],
                get_fi,he_fi,current_lookahead_map) 
            # If an exit overlap exists, set it as the leap target for Strategy 4
            if len(he_fi)!=0:
                leap_target_node=he_fi[0]


            # Toggle between standard processing and optimized strategy flow
            is_standard_flow = not (strategy_type != 0 or is_leap_strategy == 1)

        # --- Phase 3: Priority Fine-tuning for Mid-Node Matrix ---
        # Adjusting weights to force specific path selections in the Layer 3 sorter
        curr_step_idx=entry_points[cycle_idx]
        
        for j,reduction_val in enumerate(counts_tmp): 
            if reduction_val !=0: 
                # Scenario 3: Current step has unit weight; adjust preceding step's priority down
                if path_weights[ind][j] == 1: 
                    priority_dn=1-reduction_val 
                    if path_weights[ind][j-1] != 1:
                        for offset in range(reduction_val - 1):
                            target_val=elements_tmp[j-1][-1 - offset] 
                            
                            # Locate the specific path index to adjust its priority
                            sub_idx = 0
                            for val in path_indices[ind][j-1]: 
                                if val==target_val:break
                                else:sub_idx+=1
                            mid_node_matrix[cycle_idx][curr_step_idx - 1][sub_idx]+=priority_dn
                            priority_dn+=1 

                    curr_step_idx=(curr_step_idx + 1) % len(mid_node_matrix[cycle_idx])
                    priority_up=reduction_val 

                    # Adjust current step's priority up to favor the reduction path
                    for offset in range(reduction_val):
                        target_val=elements_tmp[j+1][offset] 
                        sub_idx = 0
                        for val in path_indices[ind][j+1]: 
                            if val==target_val: break
                            else: sub_idx += 1
                        mid_node_matrix[cycle_idx][curr_step_idx][sub_idx]+=priority_up
                        priority_up -= 1 

                # Scenario 2: Subsequent step has unit weight
                elif path_weights[ind][j+1]==1: 
                    priority_dn= -reduction_val 
                    for offset in range(reduction_val):
                        target_val = elements_tmp[j][-1 - offset] 
                        sub_idx = 0
                        for val in path_indices[ind][j]: 
                            if val==target_val:break
                            else: sub_idx += 1
                        mid_node_matrix[cycle_idx][curr_step_idx][sub_idx]+=priority_dn
                        priority_dn += 1 

                    curr_step_idx=(curr_step_idx + 1)%len(mid_node_matrix[cycle_idx])
                    next_step_ptr=(curr_step_idx + 1)%len(mid_node_matrix[cycle_idx])
                    priority_up = reduction_val - 1 

                    if reduction_val != 1 and path_weights[ind][j+2] != 1:
                        for offset in range(reduction_val - 1):
                            target_val = elements_tmp[j+2][offset]
                            sub_idx = 0
                            for val in path_indices[ind][j+2]:
                                if val == target_val: break
                                else: sub_idx += 1
                            mid_node_matrix[cycle_idx][next_step_ptr][sub_idx] += priority_up
                            priority_up -= 1
                # Scenario 1: Standard backward-looking alignment
                else:
                    priority_dn = -reduction_val
                    for offset in range(reduction_val):
                        target_val = elements_tmp[j][-1 - offset]
                        sub_idx = 0
                        for val in path_indices[ind][j]:
                            if val == target_val: break
                            else: sub_idx += 1
                        mid_node_matrix[cycle_idx][curr_step_idx][sub_idx] += priority_dn
                        priority_dn += 1

                    curr_step_idx = (curr_step_idx + 1) % len(mid_node_matrix[cycle_idx])
                    priority_up = reduction_val
                    for offset in range(reduction_val):
                        target_val = elements_tmp[j+1][offset]
                        sub_idx = 0
                        for val in path_indices[ind][j+1]:
                            if val == target_val: break
                            else: sub_idx += 1
                        mid_node_matrix[cycle_idx][curr_step_idx][sub_idx] += priority_up
                        priority_up -= 1
            else:
                curr_step_idx = (curr_step_idx + 1) % len(mid_node_matrix[cycle_idx])

        # --- Phase 4: Inter-Cycle Strategy Compensation ---
        if ind <= len(sorted_order)-1 and optimization_flags[ind] > 0 :
            prev_cycle_idx=sorted_order[ind-1]
            prev_step_tail = 0 if len(mid_node_matrix[prev_cycle_idx]) == 1 else entry_points[prev_cycle_idx] - 2

            # Handling Strategy 1 & 3: Aligning the tail of the previous cycle
            if optimization_flags[ind] in [1, 3]:
                if len(mid_node_matrix[prev_cycle_idx][prev_step_tail]) != 1:
                    sub_idx = 0
                    for val in path_indices[ind-1][-1]:
                        if val == reduction_indices[ind]: break
                        else: sub_idx += 1
                    mid_node_matrix[prev_cycle_idx][prev_step_tail][sub_idx] += min(mid_node_matrix[prev_cycle_idx][prev_step_tail]) - 1
                
                curr_target_step = entry_points[sorted_order[ind]]
                if len(mid_node_matrix[sorted_order[ind]][curr_target_step]) != 1:
                    sub_idx = 0
                    for val in lookahead_indices[ind-1]:
                        if val == reduction_indices[ind]: break
                        else: sub_idx += 1
                    if optimization_flags[ind] == 1:
                        mid_node_matrix[sorted_order[ind]][curr_target_step][sub_idx] += (len(mid_node_matrix[sorted_order[ind]][curr_target_step]) + 1)
            
            # Handling Strategy 2 & 4: Direct header alignment
            if optimization_flags[ind] in [2, 4]:
                curr_target_step = entry_points[sorted_order[ind]]
                if len(mid_node_matrix[sorted_order[ind]][curr_target_step]) != 1:
                    sub_idx = 0
                    for val in lookahead_indices[ind-1]:
                        if val == reduction_indices[ind]: break
                        else: sub_idx += 1
                    if optimization_flags[ind] == 2:
                        mid_node_matrix[sorted_order[ind]][curr_target_step][sub_idx] += (len(mid_node_matrix[sorted_order[ind]][curr_target_step]) + 1)

    # --- Phase 5: Final Sequence Assembly ---
    # Convert optimized mid-node priorities into final executable paths and operations
    for ind, cycle_idx in enumerate(sorted_order):
        step_ptr = entry_points[cycle_idx]
        total_steps = len(mid_node_matrix[cycle_idx])
        
        # Minimum step count adjustment for single-transition cycles
        if total_steps == 1: total_steps = 2
        
        for _ in range(total_steps - 1):
            if step_ptr >= total_steps: step_ptr = 0
            
            # Generate the specific route and fetch corresponding operation
            route = generate_state_trajectory(step_ptr, mid_node_matrix[cycle_idx][step_ptr], trajectories[cycle_idx], num_units)    
            op_gate = operation_sequences[cycle_idx][step_ptr] 
            
            final_paths.append(route)
            final_ops.append(op_gate)
            step_ptr += 1

    # Final circuit construction
    circuit= assemble_reversible_circuit(final_paths, final_ops, num_units)
    return circuit

def initialize_solution_layer(data_structure):
    """
    Resets the optimization layer to its neutral state before generating 
    a new custom solution. This clears previous search weights while 
    preserving the mandatory structural constraints.
    """
    for i in range(len(data_structure)):
        current_item = data_structure[i]
        
        if isinstance(current_item, list):
            # Check if we have reached the terminal path-selection level
            is_terminal_level = all(not isinstance(element, list) for element in current_item)
            
            if is_terminal_level:
                # 999 is a Reserved State, typically indicating a fixed or 
                # non-optimizable path. We preserve these.
                if len(current_item) == 1:
                    data_structure[i] = [999]
                else:
                    # Reset all variable path priorities to zero for the next iteration
                    data_structure[i] = [0] * len(current_item)
            else:
                # Recursive call to handle multi-dimensional solution structures
                initialize_solution_layer(current_item)
                
        else:
            # Direct reset of scalar weights, avoiding protected indices (999)
            if current_item != 999:
                data_structure[i] = 0

def analyze_bit_differences(step_idx, path_priorities, trajectory, is_flagged, num_units):
    """
    Identifies differing bit locations between two states in a trajectory.
    Returns the target bit indices, hamming distance, and state boundary nodes.
    """
    diff_locations = []
    entry_node = 0
    exit_node = 0
    
    # Check if the current step has a valid priority set (non-reserved)
    if path_priorities[0] != 999:
        # Define transition boundary nodes from the trajectory
        start_state = trajectory[step_idx]
        end_state = trajectory[step_idx + 1]
        
        # Calculate bitwise transition metrics
        h_distance = hamming_distance(start_state, end_state)
        bit_array_start = [int(b) for b in bin(start_state)[2:].zfill(num_units)]
        bit_array_end = [int(b) for b in bin(end_state)[2:].zfill(num_units)]
        
        if is_flagged:
            entry_node = start_state
            exit_node = end_state
            
        # Identify specific bit indices where transitions occur
        for i in range(len(bit_array_start)):
            if bit_array_start[i] != bit_array_end[i]:
                diff_locations.append(i)

    else:
        # RESERVED CASE: Direct transition without heuristic optimization
        start_state = trajectory[step_idx]
        end_state = trajectory[step_idx + 1]
        
        h_distance = hamming_distance(start_state, end_state)
        bit_array_start = [int(b) for b in bin(start_state)[2:].zfill(num_units)]
        bit_array_end = [int(b) for b in bin(end_state)[2:].zfill(num_units)]
        
        if is_flagged:
            entry_node = start_state
            exit_node = end_state
            
        for i in range(len(bit_array_start)):
            if bit_array_start[i] != bit_array_end[i]:
                diff_locations.append(i)

    return diff_locations, h_distance, entry_node, exit_node

def map_next_transition_bits(next_start_node, next_target_node, num_units):
    """
    Identifies differing bit indices for the first step of the next cycle.
    Used for look-ahead sequence optimization and connectivity analysis.
    """
    # Convert nodes to binary arrays representing system states
    bit_array_start = [int(b) for b in bin(next_start_node)[2:].zfill(num_units)]
    bit_array_target = [int(b) for b in bin(next_target_node)[2:].zfill(num_units)]

    transition_indices = []
    
    # Identify specific bit indices where state transitions occur
    for i in range(len(bit_array_start)):
        # Compare bits to find discrepancies (flips required)
        if bit_array_start[i] != bit_array_target[i]:
            transition_indices.append(i)
            
    return transition_indices

def reduce_sequences_standard(path_indices_list, weight_list):
    """
    Standard look-ahead reduction. Identifies and removes redundant transitions 
    by comparing current step bits with future steps.
    """
    num_steps = len(path_indices_list)
    reduction_counts = [0] * num_steps
    reduced_elements = [[] for _ in range(num_steps)]

    # Extract entry/exit metadata
    exit_node_indices = path_indices_list[-1]

    if num_steps == 1:
        return [0, 0], [[], []], exit_node_indices, []

    for i in range(num_steps - 1):
        current_step_indices = path_indices_list[i]
        matched_in_round = []
        available_weight = weight_list[i]
        
        # Adjust weight if existing reductions occupy slots
        if len(reduced_elements[i]) != 0:
            available_weight = weight_list[i] - 1 - len(reduced_elements[i])

        next_ptr = i + 1
        is_chaining = False # Tracks if we are continuing a multi-step reduction chain
        
        while next_ptr < num_steps and available_weight > 0:
            next_step_indices = path_indices_list[next_ptr]
            
            # Identify overlapping bit indices for reduction
            found_bits = [x for x in current_step_indices if x in next_step_indices and x not in matched_in_round]
            
            if found_bits:
                matched_in_round.extend(found_bits)
                temp_buffer = []
                for bit in found_bits:
                    path_indices_list[i].remove(bit)
                    path_indices_list[next_ptr].remove(bit)
                    temp_buffer.insert(0, bit)
                    reduced_elements[next_ptr].append(bit)
                    available_weight -= 1
                
                # Update current step's reduced record
                if is_chaining:
                    reduced_elements[i] = temp_buffer + reduced_elements[i]
                else:
                    reduced_elements[i].extend(temp_buffer)

            # Special Case: Preceding step synergy (Single bit flip optimization)
            if weight_list[i] == 1 and i != 0 and len(found_bits) == 1 and (reduction_counts[i-2]+1 < weight_list[i-1]):
                # Logic to check if previous step can also absorb this bit
                prev_found = [x for x in path_indices_list[i-1] if x in next_step_indices and x not in matched_in_round]
                if prev_found:
                    matched_in_round.extend(prev_found)
                    tmp_fd=[]
                    for bit in prev_found:
                        path_indices_list[i-1].remove(bit)
                        path_indices_list[next_ptr].remove(bit)
                        reduced_elements[next_ptr].append(bit)
                        
                        tmp_fd.insert(0,bit)
                        reduced_elements[i-1].extend(tmp_fd)

            # Determine if the look-ahead can proceed further
            if len(found_bits) == 1 and (next_ptr + 1) < num_steps and weight_list[next_ptr] == 1:
                next_ptr += 1
                is_chaining = True
            else:
                break
        
        reduction_counts[i] = len(matched_in_round)
        
    return reduction_counts, reduced_elements, exit_node_indices, reduced_elements[-1]

def reduce_sequences_ordered(path_indices_list, weight_list):
    """
    Performs ordered sequence reduction by tracking index offsets (ka_count).
    This ensures bit-flip consistency when integrating new reductions into existing layers.
    """
    num_steps = len(path_indices_list)
    reduction_counts = [0] * num_steps
    reduced_elements = [[] for _ in range(num_steps)]
    
    # Metadata for state tracking
    exit_boundary_indices = path_indices_list[-1]
    
    if num_steps == 1:
        reduction_counts.append(0)
        reduced_elements.append([])
        return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

    for i in range(num_steps - 1):
        current_step_indices = path_indices_list[i]
        matched_in_round = []
        available_weight = weight_list[i]
        
        # Calculate remaining weight if the current step already contains reduced bits
        if len(reduced_elements[i]) != 0:
            available_weight = weight_list[i] - 1 - len(reduced_elements[i])

        next_ptr = i + 1
        is_chaining = False 
        # Crucial: Track the current list size to maintain correct insertion offset
        insertion_offset = len(reduced_elements[i]) 
        
        while next_ptr < num_steps and available_weight > 0:
            next_step_indices = path_indices_list[next_ptr]
            found_bits = [x for x in current_step_indices if x in next_step_indices and x not in matched_in_round]
            
            if found_bits:
                matched_in_round.extend(found_bits)
                temp_buffer = [] # This is your tmp_fd
                
                for bit in found_bits:
                    path_indices_list[i].remove(bit)
                    path_indices_list[next_ptr].remove(bit)
                    temp_buffer.insert(0, bit) # Reverse order for logic consistency
                    reduced_elements[next_ptr].append(bit)
                    available_weight -= 1
                
                # Apply the specific insertion logic (ton vs. append)
                for bit in temp_buffer:
                    if is_chaining:
                        reduced_elements[i].insert(insertion_offset, bit)
                    else:
                        reduced_elements[i].append(bit)

            # Special Case: Preceding step synergy (Look-back logic)
            # Restored: (counts[i-2] + 1 < hamm[i-1]) condition equivalent
            if weight_list[i] == 1 and i != 0 and len(found_bits) == 1:
                if (reduction_counts[i-2] + 1 < weight_list[i-1]):
                    prev_found = [x for x in path_indices_list[i-1] if x in next_step_indices and x not in matched_in_round]
                    if prev_found:
                        matched_in_round.extend(prev_found)
                        temp_buffer = []
                        for bit in prev_found:
                            path_indices_list[i-1].remove(bit)
                            path_indices_list[next_ptr].remove(bit)
                            temp_buffer.insert(0, bit)
                            reduced_elements[next_ptr].append(bit)
                            available_weight -= 1
                        for bit in temp_buffer:
                            reduced_elements[i-1].append(bit)
                
            # Look-ahead chaining condition
            if len(found_bits) == 1 and (next_ptr + 1) < num_steps and weight_list[next_ptr] == 1:
                next_ptr += 1
                is_chaining = True
            else:
                break
        
        reduction_counts[i] = len(matched_in_round)
        
    return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

def reduce_sequences_targeted(path_indices_list, weight_list, common_value):
    """
    Targeted reduction for Leap Strategy alignment. Forces specific common_value 
    matching at the initial step (i=0) to ensure cross-cycle connectivity.
    """
    num_steps = len(path_indices_list)
    reduction_counts = [0] * num_steps
    reduced_elements = [[] for _ in range(num_steps)]
    exit_boundary_indices = path_indices_list[-1]
    
    if num_steps == 1:
        reduction_counts.append(0)
        reduced_elements.append([])
        return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

    for i in range(num_steps - 1):
        current_step_indices = path_indices_list[i]
        matched_in_round = []
        available_weight = weight_list[i]
        
        if len(reduced_elements[i]) != 0:
            available_weight = weight_list[i] - 1 - len(reduced_elements[i])

        next_ptr = i + 1
        is_chaining = False
        insertion_offset = len(reduced_elements[i])
        
        while next_ptr < num_steps and available_weight > 0:
            next_step_indices = path_indices_list[next_ptr]
            
            # UNIQUE KB LOGIC: Target alignment at the start of cycle
            forced_bits = []
            if i == 0:
                if common_value in next_step_indices and common_value not in matched_in_round:
                    forced_bits = [common_value]
                    matched_in_round.extend(forced_bits)
            
            found_bits = [x for x in current_step_indices if x in next_step_indices and x not in matched_in_round]
            
            # Combine forced target bits with naturally found bits
            if found_bits:
                matched_in_round.extend(found_bits)
            
            if i == 0:
                found_bits = forced_bits + found_bits
            
            
            temp_buffer = []
            for bit in found_bits:
                path_indices_list[i].remove(bit)
                path_indices_list[next_ptr].remove(bit)
                temp_buffer.insert(0, bit)
                reduced_elements[next_ptr].append(bit)
                available_weight -= 1
            
            for bit in temp_buffer:
                if is_chaining:
                    reduced_elements[i].insert(insertion_offset, bit)
                else:
                    reduced_elements[i].append(bit)

            # Special Case: Look-back synergy
            if weight_list[i] == 1 and i != 0 and len(found_bits) == 1:
                if (reduction_counts[i-2] + 1 < weight_list[i-1]):
                    prev_found = [x for x in path_indices_list[i-1] if x in next_step_indices and x not in matched_in_round]
                    if prev_found:
                        matched_in_round.extend(prev_found)
                        temp_buffer = []
                        for bit in prev_found:
                            path_indices_list[i-1].remove(bit)
                            path_indices_list[next_ptr].remove(bit)
                            temp_buffer.insert(0, bit)
                            reduced_elements[next_ptr].append(bit)
                            available_weight -= 1
                        for bit in temp_buffer:
                            reduced_elements[i-1].append(bit)
                
            if len(found_bits) == 1 and (next_ptr + 1) < num_steps and weight_list[next_ptr] == 1:
                next_ptr += 1
                is_chaining = True
            else:
                break
        
        reduction_counts[i] = len(matched_in_round)
        
    return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

def determine_transition_strategy(next_entry_node, next_second_node, 
                                 curr_entry_node, curr_exit_node, 
                                 entry_overlap_nodes, exit_overlap_nodes, 
                                 next_transition_indices):
    """
    Evaluates state connectivity between cycles to select an optimization strategy.
    Strategies 1-3 manage standard reductions, while Strategy 4 handles forced leap alignments.
    """
    
    # 'is_reduction_eligible' (formerly obe): Validation flag to ensure 
    # sequence length constraints are met for Strategies 1 & 3.
    is_reduction_eligible = False
    is_leap_enabled = 0       # Formerly ka_four (Strategy 4 flag)
    selected_strategy = 0     # Formerly inp_cycle_total (Strategy 1, 2, or 3)
    
    # Step 1: Connectivity Constraint Validation
    if len(exit_overlap_nodes) != 0:
        # Strategies 1 & 3 require multiple entry options to safely reduce
        if len(entry_overlap_nodes) > 1:
            is_reduction_eligible = True
    else:
        # If no exit overlap exists, constraints are bypassed
        is_reduction_eligible = True

    # Step 2: Distance Matrix Calculation
    # Measuring bit-flips between the boundaries of current and next cycles
    dist_next_start_curr_exit = hamming_distance(next_entry_node, curr_exit_node)
    dist_next_start_curr_start = hamming_distance(next_entry_node, curr_entry_node)
    dist_next_second_curr_exit = hamming_distance(next_second_node, curr_exit_node)
    dist_next_second_curr_start = hamming_distance(next_second_node, curr_entry_node)

    candidate_nodes = []

    # Step 3: Strategy 3 Assessment (Prioritized)
    # Check for overlapping nodes in the next transition mapping
    if len(entry_overlap_nodes) > 0:
        for node in entry_overlap_nodes:
            if node in next_transition_indices:
                candidate_nodes.append(node) 

    if (dist_next_second_curr_exit == 1 and 
        len(candidate_nodes) != 0 and 
        is_reduction_eligible):
            selected_strategy = 3 

    # Step 4: Strategy 2 Assessment
    if selected_strategy == 0 and len(exit_overlap_nodes) > 0:
        candidate_nodes = [] # Reset for Strategy 2/4 specific alignment
        
        # Verify if the primary exit node is compatible with the next transition
        if exit_overlap_nodes[0] in next_transition_indices:
            candidate_nodes.append(exit_overlap_nodes[0])
            if dist_next_start_curr_start == 1 and len(candidate_nodes) != 0:
                selected_strategy = 2

    # Step 5: Strategy 1 Assessment
    if selected_strategy == 0:
        candidate_nodes = [] # Re-scan entry overlaps
        if len(entry_overlap_nodes) > 0:
            for node in entry_overlap_nodes:
                if node in next_transition_indices:
                    candidate_nodes.append(node) 
                    
        if (dist_next_start_curr_exit == 1 and 
            len(candidate_nodes) != 0 and 
            is_reduction_eligible):
                selected_strategy = 1

    # Step 6: Strategy 4 (Leap Strategy) Assessment
    # Independent check for forced alignment when entry/exit must match
    if (dist_next_second_curr_start == 1 and 
        len(candidate_nodes) != 0 and 
        len(exit_overlap_nodes) > 0):
            is_leap_enabled = 1

    return candidate_nodes, selected_strategy, is_leap_enabled

def generate_state_trajectory(step_idx, path_priorities, trajectory, num_units):
    """
    Constructs the intermediate state sequence between two nodes.
    Flips bits based on the optimized priority sequence (path_priorities).
    """
    state_route = []
    
    # Check for non-reserved priority set
    if path_priorities[0] != 999:
        # Step 1: Sort indices based on priority values to determine flip order
        # priority_map (formerly pp) dictates which bit flips first
        sorted_elements = sorted(enumerate(path_priorities), key=lambda x: x[1], reverse=True)
        priority_map = [idx for idx, val in sorted_elements]
        
        start_node = trajectory[step_idx]
        end_node = trajectory[step_idx + 1]
        state_route.append(start_node)
        
        # Step 2: Convert to bit arrays for manipulation
        bit_array = [int(b) for b in bin(start_node)[2:].zfill(num_units)]
        target_array = [int(b) for b in bin(end_node)[2:].zfill(num_units)]
        
        # Step 3: Identify the bit locations that require flipping (Hamming indices)
        diff_locations = []
        for i in range(len(bit_array)):
            if bit_array[i] != target_array[i]:
                diff_locations.append(i)
        
        # Step 4: Execute bit-flips following the optimized priority map
        for flip_idx in priority_map:
            target_bit_pos = diff_locations[flip_idx]
            # Perform bit-flip: 0 -> 1 or 1 -> 0
            bit_array[target_bit_pos] =abs(1 - bit_array[target_bit_pos])
            
            # Convert modified bit array back to decimal state
            binary_str = ''.join(map(str, bit_array))
            decimal_state = int(binary_str, 2)
            state_route.append(decimal_state)

    else:
        # BYPASS CASE: Direct transition between start and end node
        start_node = trajectory[step_idx]
        end_node = trajectory[step_idx + 1]
        state_route.extend([start_node, end_node])

    return state_route

def assemble_reversible_circuit(state_trajectories, transition_sequence_matrix, num_bits):
    """
    Synthesizes a reversible logic circuit from state trajectories and gate transition sequences.
    Implements real-time gate cancellation (Identity Law: A * A = I) to minimize circuit depth.
    
    Args:
        state_trajectories (list): List of decimal state sequences for each transition.
        transition_sequence_matrix (list): Matrix defining the order of bit-flips (0: head-start, 1: tail-start).
        num_bits (int): Total number of bits in the system.
        
    Returns:
        tuple: (optimized_gate_list, total_raw_transitions)
    """
    raw_step_transitions = []
    
    # Step 1: Decompose state trajectories into discrete bit-flip transitions
    for i in range(len(state_trajectories)):
        head_ptr = 0
        tail_ptr = len(state_trajectories[i]) - 1
        
        total_steps = len(transition_sequence_matrix[i])
        mid_point = int(total_steps / 2)
        
        # Assemble the first half of the transition path
        # 0: Represents a flip starting from the current state (head)
        # 1: Represents a flip starting from the target state (tail)
        for j in range(mid_point):
            if transition_sequence_matrix[i][j] == 0:
                raw_step_transitions.append([state_trajectories[i][head_ptr], state_trajectories[i][head_ptr + 1]])
                head_ptr += 1
            elif transition_sequence_matrix[i][j] == 1:
                raw_step_transitions.append([state_trajectories[i][tail_ptr - 1], state_trajectories[i][tail_ptr]])
                tail_ptr -= 1
        
        # Central Pivot: The primary transition linking the head and tail paths
        if tail_ptr - head_ptr == 1 or mid_point == 0:
            raw_step_transitions.append([state_trajectories[i][head_ptr], state_trajectories[i][tail_ptr]])
            # Flip pointers to assemble the returning path correctly
            head_ptr, tail_ptr = tail_ptr, head_ptr 
        else:
            raise ValueError("Synchronization Error: Path pointers failed to meet at central pivot.")

        # Assemble the second half (return path)
        effective_limit = total_steps if mid_point != 0 else 0
        for j in range(mid_point, effective_limit):
            if transition_sequence_matrix[i][j] == 0:
                raw_step_transitions.append([state_trajectories[i][head_ptr], state_trajectories[i][head_ptr + 1]])
                head_ptr += 1
            elif transition_sequence_matrix[i][j] == 1:
                raw_step_transitions.append([state_trajectories[i][tail_ptr - 1], state_trajectories[i][tail_ptr]])
                tail_ptr -= 1

    # Step 2: Map Bit Transitions to Reversible Gates and Apply Optimization
    # Gate Encoding: 0/1 = Control Bits, 3 = Target Bit (Flip)
    optimized_gate_list = []
    
    for transition in raw_step_transitions:
        state_start, state_end = transition[0], transition[1]
        
        # Convert decimal states to binary bit arrays
        bits_start = [int(b) for b in bin(state_start)[2:].zfill(num_bits)]
        bits_end = [int(b) for b in bin(state_end)[2:].zfill(num_bits)]
        
        current_gate = []
        for bit_idx in range(num_bits):
            # Identical bits denote a Control condition; differing bits denote the Target
            if bits_start[bit_idx] == bits_end[bit_idx]:
                current_gate.append(bits_start[bit_idx])
            else:
                current_gate.append(3)
        
        # Step 3: Peephole Optimization (Gate Cancellation)
        # In reversible logic, consecutive identical gates cancel out.
        if not optimized_gate_list:
            optimized_gate_list.append(current_gate)
        else:
            if optimized_gate_list[-1] == current_gate:
                optimized_gate_list.pop() # Remove redundant gate pair
            else:
                optimized_gate_list.append(current_gate)
    
    return optimized_gate_list

def verify_circuit_logic(optimized_gates, num_bits, target_truth_table):
    """
    Validates the functional correctness of the synthesized reversible circuit.
    Simulates the circuit's operation across all possible input states (2^n).
    
    Args:
        optimized_gates (list): The list of synthesized reversible gates (Toffoli-like).
        num_bits (int): Total number of bits in the system.
        target_truth_table (list): The reference output sequence to compare against.
        
    Returns:
        int: 1 if the circuit logic matches the target truth table, 0 otherwise.
    """
    simulated_outputs = []
    
    # Generate all possible input states from 0 to 2^n - 1
    input_space = list(range(2**num_bits))
    
    for input_state in input_space:
        # Step 1: Initialize the current state bit array
        current_bits = [int(b) for b in bin(input_state)[2:].zfill(num_bits)]
        
        # Step 2: Pass the input through each logic gate in the sequence
        for gate in optimized_gates:
            target_pos = -1
            is_control_satisfied = True

            # Check Control Bit conditions (0, 1) and identify the Target Bit (3)
            for bit_idx in range(num_bits):
                gate_value = gate[bit_idx]
                
                # If the gate bit is a control (0 or 1), it must match the current state
                if (gate_value == 0 or gate_value == 1):
                    if gate_value != current_bits[bit_idx]:
                        is_control_satisfied = False
                        break
                # Identification of the target bit to be flipped (NOT operation)
                elif gate_value == 3:
                    target_pos = bit_idx
            
            # Step 3: Execute Bit-Flip (NOT) if all control conditions are met
            if target_pos != -1 and is_control_satisfied:
                current_bits[target_pos] = 1 - current_bits[target_pos]
        
        # Step 4: Convert final bit array back to decimal state
        binary_result = ''.join(map(str, current_bits))
        decimal_output = int(binary_result, 2)
        simulated_outputs.append(decimal_output)
    
    # Step 5: Final Integrity Comparison
    # Check if the entire simulated output set matches the target truth table
    if simulated_outputs == target_truth_table:
        return 1
    else:
        return 0
    
  