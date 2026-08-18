"""
合併 exp_5000_1 ~ exp_5000_4 的 AE-QTS 結果。

每個資料夾的 xlsx 內含 25 次 trial，四個資料夾合併後為 100 次，
重新計算 Average_Convergence / Std_Deviation，輸出新的 xlsx 與收斂圖。

輸出：exp_5000_merged/{bit}_bit/AE-QTS_Results/
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
SOURCE_FOLDERS = ["exp_5000_1", "exp_5000_2", "exp_5000_3", "exp_5000_4"]
OUTPUT_ROOT = "exp_5000_merged"
BITS = [6, 7, 8, 9, 10]
ALGO = "AE-QTS"
RESULT_DIR = f"{ALGO}_Results"

AVG_ROW = "Average_Convergence"
STD_ROW = "Std_Deviation"
TIME_COL = "Execution_Time(s)"

# 每個 sheet 對應的收斂圖設定：{sheet 名稱: (輸出檔名後綴, y 軸標籤, 圖表標題, 是否標註最佳解)}
PLOT_SHEETS = {
    "Total_Gate_Count": ("avg_convergence", "Average Gate Count", "Total Gate Count Convergence", True),
    "Mode_Count": ("mode_count_convergence", "Mode Count", "Mode Count Convergence", False),
}
# 四條 entropy 曲線畫在同一張圖
ENTROPY_SHEETS = ["Entropy_Q1", "Entropy_Q2", "Entropy_Q3", "Entropy_Q4"]


def find_source_file(folder, bit):
    """在 folder/{bit}_bit/AE-QTS_Results/ 底下找對應的 xlsx。"""
    pattern = os.path.join(folder, f"{bit}_bit", RESULT_DIR, "*.xlsx")
    files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith("~$")]
    return files[0] if files else None


def load_trials(path, sheet_name):
    """讀取單一 sheet，只保留 Trial_* 的資料列（丟掉統計列）。"""
    df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    df.index = df.index.astype(str).str.strip()
    trial_mask = df.index.str.startswith("Trial_")
    return df[trial_mask]


def merge_bit(bit):
    """合併同一個位元數的四份資料，回傳 {sheet: DataFrame(100 trials)}。"""
    sources = []
    for folder in SOURCE_FOLDERS:
        path = find_source_file(folder, bit)
        if path is None:
            print(f"  [warn] {folder}/{bit}_bit 找不到 xlsx，跳過")
            continue
        sources.append((folder, path))

    if not sources:
        return None, None

    sheet_names = pd.ExcelFile(sources[0][1]).sheet_names
    merged = {}

    for sheet in sheet_names:
        parts = []
        for folder, path in sources:
            df = load_trials(path, sheet)
            parts.append(df)
            print(f"  {sheet:<18} <- {folder}: {len(df)} trials")
        combined = pd.concat(parts, axis=0, ignore_index=True)
        # 重新編號 Trial_1 ~ Trial_100
        combined.index = [f"Trial_{i + 1}" for i in range(len(combined))]
        merged[sheet] = combined

    return merged, [p for _, p in sources]


def add_statistics(df):
    """在 100 次 trial 後面補上平均與標準差列。"""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame(
        [numeric.mean(axis=0), numeric.std(axis=0, ddof=0)],
        index=[AVG_ROW, STD_ROW],
        columns=df.columns,
    )
    return pd.concat([df, stats], axis=0)


def gen_columns(df):
    """取出 Gen_* 欄位（排除 Execution_Time）。"""
    return [c for c in df.columns if str(c).startswith("Gen_")]


def plot_convergence(avg, std, title, ylabel, save_path, n_trials, best_curve=None, finals=None):
    """
    best_curve: 100 次中最佳那一次的收斂曲線（可選）
    finals    : 100 次各自的最終解，用來標註平均最佳解與 100 次中最好的解
    """
    gens = np.arange(1, len(avg) + 1)
    plt.figure(figsize=(11, 6.5))
    plt.plot(gens, avg, color="#F40E0E", linewidth=2.0, label=f"Mean of {n_trials} runs")
    plt.fill_between(gens, avg - std, avg + std, color="#F40E0E", alpha=0.15, label="±1 Std. Dev.")

    if best_curve is not None:
        plt.plot(gens, best_curve, color="#1F77B4", linewidth=1.6, linestyle="--",
                 label=f"Best of {n_trials} runs")

    if finals is not None and len(finals) > 0:
        mean_best = float(np.mean(finals))
        std_best = float(np.std(finals))
        overall_best = float(np.min(finals))
        best_count = int(np.sum(finals == np.min(finals)))
        last_gen = len(avg)

        # 平均最佳解 水平參考線 + 標註
        plt.axhline(mean_best, color="#F40E0E", linewidth=1.0, linestyle=":", alpha=0.8)
        plt.annotate(f"Mean best = {mean_best:.2f} ± {std_best:.2f}",
                     xy=(last_gen, mean_best), xytext=(-12, 10),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=10, fontweight="bold", color="#F40E0E",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F40E0E", alpha=0.85))

        # 100 次中最好的解 水平參考線 + 標註
        plt.axhline(overall_best, color="#1F77B4", linewidth=1.0, linestyle=":", alpha=0.8)
        plt.annotate(f"Best = {overall_best:.0f}  (hit {best_count}/{n_trials})",
                     xy=(last_gen, overall_best), xytext=(-12, -14),
                     textcoords="offset points", ha="right", va="top",
                     fontsize=10, fontweight="bold", color="#1F77B4",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1F77B4", alpha=0.85))

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_entropy(merged, title, save_path, n_trials):
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]
    plt.figure(figsize=(11, 6.5))
    plotted = False
    for sheet, color in zip(ENTROPY_SHEETS, colors):
        if sheet not in merged:
            continue
        df = merged[sheet]
        cols = gen_columns(df)
        avg = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy()
        plt.plot(np.arange(1, len(avg) + 1), avg, color=color, linewidth=1.8, label=sheet)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel(f"Average Entropy ({n_trials} runs)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    for bit in BITS:
        print(f"\n=== {bit}-bit ===")
        merged, sources = merge_bit(bit)
        if merged is None:
            print("  跳過（無資料）")
            continue

        n_trials = len(next(iter(merged.values())))
        out_dir = os.path.join(OUTPUT_ROOT, f"{bit}_bit", RESULT_DIR)
        os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(sources[0]))[0]  # 例如 AE-QTS_6_1
        xlsx_path = os.path.join(out_dir, f"{base_name}.xlsx")

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet, df in merged.items():
                add_statistics(df).to_excel(writer, sheet_name=sheet)
        print(f"  -> {xlsx_path}（{n_trials} trials）")

        # 各項收斂圖
        for sheet, (suffix, ylabel, title, mark_best) in PLOT_SHEETS.items():
            if sheet not in merged:
                continue
            df = merged[sheet]
            cols = gen_columns(df)
            numeric = df[cols].apply(pd.to_numeric, errors="coerce")

            best_curve = None
            finals = None
            if mark_best:
                # 以最後一代的結果作為每次 run 的最佳解（gate count 越小越好）
                finals = numeric[cols[-1]].to_numpy(dtype=float)
                best_curve = numeric.iloc[int(np.argmin(finals))].to_numpy(dtype=float)

            plot_convergence(
                numeric.mean(axis=0).to_numpy(),
                numeric.std(axis=0, ddof=0).to_numpy(),
                f"{title} - {base_name} ({n_trials} runs)",
                ylabel,
                os.path.join(out_dir, f"{base_name}_{suffix}.png"),
                n_trials,
                best_curve=best_curve,
                finals=finals,
            )

        plot_entropy(
            merged,
            f"Entropy Convergence - {base_name} ({n_trials} runs)",
            os.path.join(out_dir, f"{base_name}_entropy_convergence.png"),
            n_trials,
        )

        # 平均執行時間
        if TIME_COL in merged["Total_Gate_Count"].columns:
            times = pd.to_numeric(merged["Total_Gate_Count"][TIME_COL], errors="coerce")
            print(f"  平均執行時間：{times.mean():.2f}s")

        final = pd.to_numeric(
            merged["Total_Gate_Count"][gen_columns(merged["Total_Gate_Count"])[-1]],
            errors="coerce",
        )
        print(f"  平均最佳解：{final.mean():.2f} ± {final.std(ddof=0):.2f}")
        print(f"  {n_trials} 次中最好的解：{final.min():.0f}"
              f"（出現 {int((final == final.min()).sum())} 次）")

    print(f"\n[System] 全部輸出於：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
