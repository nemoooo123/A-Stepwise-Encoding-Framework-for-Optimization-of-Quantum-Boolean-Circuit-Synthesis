import os
import re
import matplotlib.pyplot as plt

# --- Configuration: Root folder where experiment data is stored ---
root_folder = "exp"

# 1. Define algorithm styles and folder mapping
# Change labels, colors, or line widths here for the plot
algorithms = {
    "AE-QTS": {"folder": "AE-QTS_Results", "label": "Ours (AE-QTS)", "color": "#F40E0E", "lw": 2.5, "marker": None},
    "QTS":    {"folder": "QTS_Results",    "label": "QTS",           "color": "#FDAA04", "lw": 2.0, "marker": None},
    "PSO":    {"folder": "PSO_Results",    "label": "PSO",           "color": "#00FF44", "lw": 2.0, "marker": None},
    "DE":     {"folder": "DE_Results",     "label": "DE",            "color": "#2B00FF", "lw": 2.0, "marker": None},
    "TS":     {"folder": "TS_Results",     "label": "Tabu Search",   "color": "gray",    "lw": 1.5, "marker": None}
}

# 2. Define task ranges (IMPORTANT: Update this when adding new problems)
# Format: { num_bits: [start_index, end_index] }
tasks = {
    3: [1, 13],
    4: [1, 9],
    5: [1, 5],
    6: [1, 5],
    7: [1, 5],
    8: [1, 5],
    9: [1, 5],
    10: [1, 5],
    11: [1, 5],
    12: [1, 5],
    13: [1, 5]
}

# 3. Create output directory for PDF plots
output_dir = "Comparison_PDF_Plots"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# --- Start batch plotting process ---
for bit, range_info in tasks.items():
    start_id, end_id = range_info
    bit_folder = os.path.join(root_folder, f"{bit}_bit")
    
    for prob_id in range(start_id, end_id + 1):
        target_problem = f"{bit}_{prob_id}"
        results_summary = {}
        
        print(f"Generating PDF from {root_folder}: {target_problem}...", end=" ")

        for algo_name, config in algorithms.items():
            file_path = os.path.join(bit_folder, config["folder"], f"{algo_name}_{target_problem}.txt")
            
            if os.path.exists(file_path):
                try:
                    # Files are saved in UTF-16 encoding
                    with open(file_path, 'r', encoding='utf-16') as f:
                        content = f.read()
                    
                    # --- Regex parsing for new output format ---
                    # 1. Match Final Result (Mean ± Std Dev)
                    final_match = re.search(r"Final Result \(Gen \d+\): ([\d\.]+) ± ([\d\.]+)", content)
                    f_avg = final_match.group(1) if final_match else "N/A"
                    f_std = final_match.group(2) if final_match else "0.00"

                    # 2. Match Average Time per Experiment
                    avg_time_match = re.search(r"Average Time per Experiment: ([\d\.]+)s", content)
                    a_time = avg_time_match.group(1) if avg_time_match else "N/A"

                    # 3. Match Average Gates convergence sequence
                    avg_gates_match = re.search(r"Average Gates = \[(.*?)\]", content, re.DOTALL)

                    if avg_gates_match:
                        raw_gates = avg_gates_match.group(1).replace('\n', ' ')
                        convergence = [float(x) for x in raw_gates.split() if x.strip() and x != '.']
                        
                        results_summary[algo_name] = {
                            "final_avg": f_avg,
                            "final_std": f_std,
                            "time": a_time,
                            "convergence": convergence,
                            "config": config
                        }
                except Exception as e:
                    print(f"Error reading {algo_name}: {e}")
                    continue

        # --- 4. Plotting using parsed parameters ---
        if results_summary:
            plt.figure(figsize=(10, 6))
            for algo, data in results_summary.items():
                conf = data["config"]
                # Legend format: Name (Avg ± SD, Time s)
                label_str = f"{conf['label']} ({data['final_avg']}±{data['final_std']}, {data['time']}s)"
                
                plt.plot(data['convergence'], 
                         label=label_str,
                         color=conf['color'], 
                         linewidth=conf['lw'], 
                         marker=conf['marker'], 
                         markersize=4 if conf['marker'] else 0, 
                         markevery=100)

            plt.title(f"Quantum Circuit Synthesis Comparison - Problem {target_problem}", fontsize=14)
            plt.xlabel("Generation", fontsize=12)
            plt.ylabel("Average Gate Count", fontsize=12)
            plt.legend(loc='upper right', frameon=True, shadow=True, prop={'size': 9})
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"Comparison_{target_problem}.pdf")
            plt.savefig(save_path)
            plt.close()
            print("Done")
        else:
            print("Skipped (No data found)")

print(f"\n[System] All PDF plots have been saved to: {output_dir}")