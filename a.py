import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
num_bits = 6
problem_idx = 1
algo_names = ["AE-QTS", "DE"]   # folder is "{algo_name}_Results", file is "{algo_name}_{num_bits}_{problem_idx}.xlsx"


def load_convergence(algo_name, sheet_name):
    xlsx_path = os.path.join("exp", f"{num_bits}_bit", f"{algo_name}_Results", f"{algo_name}_{num_bits}_{problem_idx}.xlsx")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    gen_cols = [c for c in df.columns if c.startswith("Gen_")]
    return df.loc["Average_Convergence", gen_cols].astype(float).tolist()


# One separate figure per algorithm, each showing its Total/Unique gate count convergence
for algo_name in algo_names:
    total_convergence = load_convergence(algo_name, "Total_Gate_Count")
    unique_convergence = load_convergence(algo_name, "Unique_Gate_Count")

    plt.figure()
    plt.plot(range(1, len(total_convergence) + 1), total_convergence, label="avg total gate count", color="blue")
    plt.plot(range(1, len(unique_convergence) + 1), unique_convergence, label="avg unique gate count", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Gate Count")
    plt.title(f"{algo_name} Averaged Convergence Curve (30 experiments)")
    plt.legend()

plt.show()
