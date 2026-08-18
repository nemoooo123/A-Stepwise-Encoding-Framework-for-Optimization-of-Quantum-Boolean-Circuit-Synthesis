"""
合併「固定」資料夾內 6 位元的四個方法（每個方法 exp固定解N_1 + exp固定解N_2，各 50 次 → 100 次），
再加上 exp_5000_merged 的 6 位元 AE-QTS（100 次），畫成 5 條線的收斂比較圖。

輸出：Fixed_6bit/
  Fixed_{N}_merged.xlsx      各方法合併後的 100 次數據（含重算的平均/標準差列）
  6bit_fixed_comparison.png  5 條收斂曲線（含平均最佳解與 100 次最好解標註）
  fixed_summary.xlsx         5 個方法的統計總表
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
FIXED_ROOT = "固定"
MERGED_XLSX = os.path.join("exp_5000_merged", "6_bit", "AE-QTS_Results", "AE-QTS_6_1.xlsx")
OUTPUT_DIR = "Fixed_6bit"
REL_PATH = os.path.join("6_bit", "AE-QTS_Results", "AE-QTS_6_1.xlsx")

AVG_ROW = "Average_Convergence"
STD_ROW = "Std_Deviation"

# 四個方法：資料夾前綴 -> (圖例名稱, 顏色)。想改圖例文字直接改這裡。
FIXED_METHODS = {
    1: ("Fixed-1", "#1F77B4"),
    2: ("Fixed-2", "#2CA02C"),
    3: ("Fixed-3", "#FF7F0E"),
    4: ("Fixed-4", "#9467BD"),
}
BASELINE_LABEL = "AE-QTS (exp_5000_merged)"
BASELINE_COLOR = "#F40E0E"


def load_trials(path, sheet_name="Total_Gate_Count"):
    """讀單一 xlsx 的 Trial_* 資料列（丟掉統計列）。"""
    df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df = df[df.index.str.startswith("Trial_")]
    return df


def gen_columns(df):
    return [c for c in df.columns if str(c).startswith("Gen_")]


def merge_method(n):
    """合併 exp固定解N_1 + exp固定解N_2 的所有 sheet，回傳 {sheet: DataFrame}。"""
    paths = []
    for part in (1, 2):
        p = os.path.join(FIXED_ROOT, f"exp固定解{n}_{part}", REL_PATH)
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"  [warn] 找不到 {p}")
    if not paths:
        return None

    sheet_names = pd.ExcelFile(paths[0]).sheet_names
    merged = {}
    for sheet in sheet_names:
        parts = [load_trials(p, sheet) for p in paths]
        combined = pd.concat(parts, axis=0, ignore_index=True)
        combined.index = [f"Trial_{i + 1}" for i in range(len(combined))]
        merged[sheet] = combined
    print(f"  Fixed-{n}: " + " + ".join(str(len(load_trials(p))) for p in paths)
          + f" = {len(merged['Total_Gate_Count'])} trials")
    return merged


def add_statistics(df):
    numeric = df.apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame([numeric.mean(axis=0), numeric.std(axis=0, ddof=0)],
                         index=[AVG_ROW, STD_ROW], columns=df.columns)
    return pd.concat([df, stats], axis=0)


def stats_of(curves):
    finals = curves[:, -1]
    return {
        "runs": len(curves),
        "generations": curves.shape[1],
        "mean_best": float(finals.mean()),
        "std_best": float(finals.std(ddof=0)),
        "best": float(finals.min()),
        "worst": float(finals.max()),
        "best_hit": int((finals == finals.min()).sum()),
        "median": float(np.median(finals)),
    }


def plot_all(series, save_path):
    """series: [(label, color, curves)]，curves 為 (runs, gens) 陣列。"""
    plt.figure(figsize=(12.5, 7.5))
    ax = plt.gca()

    for label, color, curves in series:
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=0)
        gens = np.arange(1, len(mean) + 1)
        s = stats_of(curves)
        lw = 2.4 if label == BASELINE_LABEL else 1.8
        ls = "-" if label == BASELINE_LABEL else "--"
        ax.plot(gens, mean, color=color, lw=lw, ls=ls,
                label=f"{label}  (mean {s['mean_best']:.2f}±{s['std_best']:.2f}, "
                      f"best {s['best']:.0f}×{s['best_hit']})")
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.10)
        ax.axhline(s["mean_best"], color=color, lw=0.8, ls=":", alpha=0.7)
        ax.axhline(s["best"], color=color, lw=0.8, ls=(0, (1, 4)), alpha=0.5)

    # 右側標註，依平均最佳解排序後往上錯開避免重疊
    last_gen = max(c.shape[1] for _, _, c in series)
    for i, (label, color, curves) in enumerate(
            sorted(series, key=lambda t: stats_of(t[2])["mean_best"])):
        s = stats_of(curves)
        ax.annotate(f"{label}: mean {s['mean_best']:.2f} / best {s['best']:.0f}",
                    xy=(last_gen, s["mean_best"]),
                    xytext=(-10, 10 + 15 * i), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=color, alpha=0.85))

    ax.set_title("6-bit Convergence Comparison (100 runs each, 5000 generations)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Gate Count (best-so-far)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, shadow=True, prop={"size": 9})
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    series = []
    summary = []

    print("=== 合併「固定」四個方法 ===")
    for n, (label, color) in FIXED_METHODS.items():
        merged = merge_method(n)
        if merged is None:
            continue

        xlsx_path = os.path.join(OUTPUT_DIR, f"Fixed_{n}_merged.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet, df in merged.items():
                add_statistics(df).to_excel(writer, sheet_name=sheet)
        print(f"    -> {xlsx_path}")

        gc = merged["Total_Gate_Count"]
        curves = gc[gen_columns(gc)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        series.append((label, color, curves))
        summary.append({"method": label, **stats_of(curves)})

    print("\n=== 讀取 exp_5000_merged 的 6-bit AE-QTS ===")
    if os.path.exists(MERGED_XLSX):
        df = load_trials(MERGED_XLSX)
        curves = df[gen_columns(df)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        series.append((BASELINE_LABEL, BASELINE_COLOR, curves))
        summary.append({"method": BASELINE_LABEL, **stats_of(curves)})
        print(f"  {BASELINE_LABEL}: {curves.shape[0]} trials x {curves.shape[1]} gens")
    else:
        print(f"  [warn] 找不到 {MERGED_XLSX}（請先執行 merge_5000.py）")

    if not series:
        print("沒有可用資料")
        return

    png = os.path.join(OUTPUT_DIR, "6bit_fixed_comparison.png")
    plot_all(series, png)
    print(f"\n[System] 收斂圖：{png}")

    df = pd.DataFrame(summary).set_index("method")
    out = os.path.join(OUTPUT_DIR, "fixed_summary.xlsx")
    df.to_excel(out)
    print(f"[System] 總表：{out}")
    print(df.to_string())


if __name__ == "__main__":
    main()
