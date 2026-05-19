import os
import re
import matplotlib.pyplot as plt
import pandas as pd  

# --- Configuration: Root folder where experiment data is stored ---
root_folder = "exp"

# 1. 精確對齊的 9 個演算法配置
algorithms = {
    "AE-QTS": {"folder": "AE-QTS_Results", "label": "Ours (AE-QTS)", "color": "#F40E0E", "lw": 2.5},
    "QTS":    {"folder": "QTS_Results",    "label": "QTS",           "color": "#FDAA04", "lw": 1.8},
    "QEA":    {"folder": "QEA_Results",    "label": "QEA",           "color": "#1F77B4", "lw": 1.8},
    "GA":     {"folder": "GA_Results",     "label": "GA",            "color": "#00FF44", "lw": 1.8},
    "DE":     {"folder": "DE_Results",     "label": "DE",            "color": "#2B00FF", "lw": 1.8},
    "TS":     {"folder": "TS_Results",     "label": "Tabu Search",   "color": "#9467BD", "lw": 1.5},
    "PSO":    {"folder": "PSO_Results",    "label": "PSO",           "color": "#E377C2", "lw": 1.5},
    "WOA":    {"folder": "WOA_Results",    "label": "WOA",           "color": "#BCBD22", "lw": 1.5},
    "ABC":    {"folder": "ABC_Results",    "label": "ABC",           "color": "gray",    "lw": 1.5}
}

# 2. Define task ranges
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
    
    if not os.path.exists(bit_folder):
        continue
        
    for prob_id in range(start_id, end_id + 1):
        target_problem = f"{bit}_{prob_id}"  # 例如 "6_1"
        results_summary = {}
        
        print(f"Generating PDF for {target_problem}...", end=" ")

        for algo_name, config in algorithms.items():
            target_dir = os.path.join(bit_folder, config["folder"])
            
            if not os.path.exists(target_dir):
                continue
            
            # 使用模糊尋找鎖定該任務的 .xlsx 檔案
            matched_file = None
            try:
                all_files = os.listdir(target_dir)
                for file_name in all_files:
                    name_lower = file_name.lower()
                    is_valid_xlsx = file_name.endswith('.xlsx') and not file_name.startswith('~$')
                    contains_algo = algo_name.lower() in name_lower
                    contains_prob = (target_problem in name_lower) or (target_problem.replace('_', '-') in name_lower)
                    
                    if is_valid_xlsx and contains_algo and contains_prob:
                        matched_file = os.path.join(target_dir, file_name)
                        break
            except:
                continue
            
            # 找到橫向 Excel 檔案後開始提取
            if matched_file:
                try:
                    df = pd.read_excel(matched_file, header=None)
                    
                    # 透過文字比對動態定位「平均收斂列」與「標準差列」
                    df_col0_clean = df[0].astype(str).str.strip()
                    
                    avg_mask = df_col0_clean == 'Average_Convergence'
                    std_mask = df_col0_clean == 'Std_Deviation'
                    
                    avg_row_idx = df[avg_mask].index[0] if avg_mask.any() else 3
                    std_row_idx = df[std_mask].index[0] if std_mask.any() else 4
                    
                    # 【核心修正】：橫向截取 1 到 1000 欄的收斂數據
                    raw_convergence_data = df.iloc[avg_row_idx, 1:1001]
                    numeric_convergence = pd.to_numeric(raw_convergence_data, errors='coerce').dropna()
                    convergence = numeric_convergence.tolist()
                    
                    if convergence:
                        # 1. 最終均值 (最後一代的值)
                        f_avg = f"{float(convergence[-1]):.2f}"
                        
                        # 2. 最終標準差 (從標準差列對應的位置抓取)
                        try:
                            raw_std = df.iloc[std_row_idx, len(convergence)]
                            f_std = f"{float(raw_std):.2f}"
                        except:
                            f_std = "0.00"
                        
                        # 3. 平均執行時間 (在平均列的第 1001 欄位)
                        try:
                            if df.shape[1] > 1001:
                                raw_time = df.iloc[avg_row_idx, 1001]
                                a_time = f"{float(raw_time):.2f}"
                            else:
                                a_time = "N/A"
                        except:
                            a_time = "N/A"
                        
                        results_summary[algo_name] = {
                            "final_avg": f_avg,
                            "final_std": f_std,
                            "time": a_time,
                            "convergence": convergence,
                            "config": config
                        }
                except Exception as e:
                    # 如果讀取 Excel 失敗，自動嘗試讀取同名的舊版 .txt 備用路徑
                    pass
            
            # --- 備用方案 B: 如果沒 Excel 則尋找純文字格式 (.txt) ---
            else:
                txt_path = os.path.join(target_dir, f"{algo_name}_{target_problem}.txt")
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-16') as f:
                            content = f.read()
                        
                        final_match = re.search(r"Final Result \(Gen \d+\): ([\d\.]+) ± ([\d\.]+)", content)
                        f_avg = final_match.group(1) if final_match else "N/A"
                        f_std = final_match.group(2) if final_match else "0.00"

                        avg_time_match = re.search(r"Average Time per Experiment: ([\d\.]+)s", content)
                        a_time = avg_time_match.group(1) if avg_time_match else "N/A"

                        avg_gates_match = re.search(r"Average Gates = \[(.*?)\]", content, re.DOTALL)
                        if avg_gates_match:
                            raw_gates = avg_gates_match.group(1).replace('\n', ' ')
                            convergence = [float(x) for x in raw_gates.split() if x.strip() and x != '.']
                            
                            results_summary[algo_name] = {
                                "final_avg": f_avg, "final_std": f_std, "time": a_time,
                                "convergence": convergence, "config": config
                            }
                    except:
                        continue

        # --- 4. 繪製 9 個演算法的大車拼圖表 ---
        if results_summary:
            plt.figure(figsize=(11.5, 7.5)) # 微調畫布大小以容納完整豐富的圖例
            for algo, data in results_summary.items():
                conf = data["config"]
                # 經典傳承格式：演算法名稱 (均值 ± 標準差, 時間s)
                label_str = f"{conf['label']} ({data['final_avg']}±{data['final_std']}, {data['time']}s)"
                
                plt.plot(data['convergence'], 
                         label=label_str,
                         color=conf['color'], 
                         linewidth=conf['lw'], 
                         markevery=100)

            plt.title(f"Quantum Circuit Synthesis Comparison - Problem {target_problem}", fontsize=14, fontweight='bold')
            plt.xlabel("Generation", fontsize=12)
            plt.ylabel("Average Gate Count", fontsize=12)
            plt.legend(loc='upper right', frameon=True, shadow=True, prop={'size': 8.5})
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"Comparison_{target_problem}.pdf")
            plt.savefig(save_path)
            plt.close()
            print("Done")
        else:
            print("Skipped (No valid data found)")

print(f"\n[System] All PDF plots have been successfully saved to: {output_dir}")