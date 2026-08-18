"""
把 DE_5000_merged（DE 100 次）與已整理好的 新增資料夾/Combined_5000
（AE-QTS / AE-QTS-clamp / AE-QTS-repair，各 100 次）放在一起比較。

Combined_5000 的 xlsx 只保留了平均/標準差/最佳收斂曲線與 100 次最終解，
因此本腳本以「曲線 + 最終解」為單位做統計，不需要原始 100 條曲線。

輸出：Combined_5000_with_DE/
  {bit}bit_with_DE.xlsx        四種方法的平均/標準差/最佳收斂曲線、100 次最終解、統計摘要
  {bit}bit_with_DE.png         四種方法的收斂比較圖
  DE_only_convergence.png      DE 各位元合併後的收斂圖（100 次平均 ±1 std）
  overall_summary_with_DE.xlsx 所有位元的總表
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
DE_ROOT = "DE_5000_merged"                      # merge_de_5000.py 的輸出
COMBINED_DIR = os.path.join("新增資料夾", "Combined_5000")  # AE-QTS / clamp / repair 100 次
OUTPUT_DIR = "Combined_5000_with_DE"
BITS = [6, 7, 8, 9, 10]
AE_METHODS = ["AE-QTS", "AE-QTS-clamp", "AE-QTS-repair"]

METHOD_STYLE = {
    "AE-QTS":        {"color": "#F40E0E", "lw": 2.2, "ls": "-"},
    "AE-QTS-clamp":  {"color": "#1F77B4", "lw": 1.8, "ls": "--"},
    "AE-QTS-repair": {"color": "#2CA02C", "lw": 1.8, "ls": "-."},
    "DE":            {"color": "#FF7F0E", "lw": 2.0, "ls": "-"},
}


# ---------- 讀取 ----------

def load_de(bit):
    """讀 DE_5000_merged 的 100 次收斂曲線，回傳 (curves, mean_time_s)。"""
    pattern = os.path.join(DE_ROOT, f"{bit}_bit", "DE_Results", "*.xlsx")
    files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith("~$")]
    if not files:
        return None, None

    df = pd.read_excel(files[0], sheet_name="Total_Gate_Count", index_col=0)
    df.index = df.index.astype(str).str.strip()
    trials = df[df.index.str.startswith("Trial_")]

    gen_cols = [c for c in trials.columns if str(c).startswith("Gen_")]
    curves = trials[gen_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    mean_time = None
    if "Execution_Time(s)" in trials.columns:
        t = pd.to_numeric(trials["Execution_Time(s)"], errors="coerce")
        mean_time = float(t.mean())
    return curves, mean_time


def load_combined(bit):
    """讀 Combined_5000 的三種 AE-QTS 方法，回傳 {method: {...}}。"""
    path = os.path.join(COMBINED_DIR, f"{bit}bit_combined.xlsx")
    if not os.path.exists(path):
        return {}

    curves = pd.read_excel(path, sheet_name="Mean_Curves", index_col=0)
    finals = pd.read_excel(path, sheet_name="Final_Solutions", index_col=0)
    summary = pd.read_excel(path, sheet_name="Summary", index_col=0)

    out = {}
    for method in AE_METHODS:
        if f"{method}_Mean" not in curves.index or method not in finals.columns:
            continue
        f = pd.to_numeric(finals[method], errors="coerce").dropna().to_numpy(dtype=float)
        time_s = None
        if method in summary.index and "total_time_s" in summary.columns:
            v = summary.loc[method, "total_time_s"]
            time_s = None if pd.isna(v) else float(v)
        out[method] = {
            "mean": curves.loc[f"{method}_Mean"].to_numpy(dtype=float),
            "std": curves.loc[f"{method}_Std"].to_numpy(dtype=float),
            "best_curve": curves.loc[f"{method}_Best"].to_numpy(dtype=float),
            "finals": f,
            "time_s": time_s,
        }
    return out


# ---------- 統計與繪圖 ----------

def stats_of(info):
    f = info["finals"]
    return {
        "runs": len(f),
        "generations": len(info["mean"]),
        "mean_best": float(f.mean()),
        "std_best": float(f.std(ddof=0)),
        "best": float(f.min()),
        "worst": float(f.max()),
        "best_hit": int((f == f.min()).sum()),
        "median": float(np.median(f)),
        "time_s_per_run": info["time_s"],
    }


def plot_comparison(bit, data, save_path):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    for method, info in data.items():
        style = METHOD_STYLE[method]
        mean, std = info["mean"], info["std"]
        gens = np.arange(1, len(mean) + 1)
        s = info["stats"]

        label = (f"{method}  (mean {s['mean_best']:.2f}±{s['std_best']:.2f}, "
                 f"best {s['best']:.0f}×{s['best_hit']})")
        ax.plot(gens, mean, color=style["color"], lw=style["lw"], ls=style["ls"], label=label)
        ax.fill_between(gens, mean - std, mean + std, color=style["color"], alpha=0.12)

        ax.axhline(s["mean_best"], color=style["color"], lw=0.9, ls=":", alpha=0.75)
        ax.axhline(s["best"], color=style["color"], lw=0.9, ls=(0, (1, 3)), alpha=0.55)

    # 標註集中在右側，依平均最佳解排序後錯開避免重疊
    last_gen = max(len(info["mean"]) for info in data.values())
    for i, (method, info) in enumerate(
            sorted(data.items(), key=lambda kv: kv[1]["stats"]["mean_best"])):
        s = info["stats"]
        color = METHOD_STYLE[method]["color"]
        ax.annotate(f"{method}: mean {s['mean_best']:.2f} / best {s['best']:.0f}",
                    xy=(last_gen, s["mean_best"]),
                    xytext=(-10, 12 + 16 * i), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, alpha=0.85))

    ax.set_title(f"{bit}-bit Convergence Comparison — AE-QTS variants vs DE "
                 f"(100 runs each, 5000 generations)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Gate Count (best-so-far)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, shadow=True, prop={"size": 9})
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_de_only(de_data, save_path):
    """DE 自己的收斂圖：各位元一個子圖（100 次平均 ±1 std）。"""
    bits = sorted(de_data)
    if not bits:
        return
    ncol = 3
    nrow = int(np.ceil(len(bits) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)

    for ax, bit in zip(axes.ravel(), bits):
        info = de_data[bit]
        mean, std = info["mean"], info["std"]
        gens = np.arange(1, len(mean) + 1)
        s = info["stats"]
        c = METHOD_STYLE["DE"]["color"]
        ax.plot(gens, mean, color=c, lw=1.8, label=f"Mean of {s['runs']} runs")
        ax.fill_between(gens, mean - std, mean + std, color=c, alpha=0.15, label="±1 Std. Dev.")
        ax.plot(gens, info["best_curve"], color="#1F77B4", lw=1.2, ls="--",
                label=f"Best of {s['runs']} runs")
        ax.set_title(f"DE {bit}-bit  (mean {s['mean_best']:.2f}±{s['std_best']:.2f}, "
                     f"best {s['best']:.0f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Gate Count")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(bits):]:
        ax.axis("off")

    fig.suptitle("DE Convergence — exp5000_1~4 merged (100 runs, 5000 generations)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------- 主流程 ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall = []
    de_data = {}

    for bit in BITS:
        print(f"\n=== {bit}-bit ===")
        data = load_combined(bit)
        for method, info in data.items():
            print(f"  {method:<15}: {len(info['finals'])} runs x {len(info['mean'])} gens"
                  f"  (from {COMBINED_DIR})")
        if not data:
            print(f"  [warn] 找不到 {COMBINED_DIR}/{bit}bit_combined.xlsx")

        curves, mean_time = load_de(bit)
        if curves is None:
            print(f"  [warn] 找不到 {DE_ROOT}/{bit}_bit 的 DE 合併資料")
        else:
            data["DE"] = {
                "mean": curves.mean(axis=0),
                "std": curves.std(axis=0, ddof=0),
                "best_curve": curves.min(axis=0),
                "finals": curves[:, -1],
                "time_s": mean_time,
            }
            print(f"  {'DE':<15}: {curves.shape[0]} runs x {curves.shape[1]} gens"
                  f"  (from {DE_ROOT})")

        if not data:
            continue

        for info in data.values():
            info["stats"] = stats_of(info)
        if "DE" in data:
            de_data[bit] = data["DE"]

        # --- xlsx ---
        xlsx_path = os.path.join(OUTPUT_DIR, f"{bit}bit_with_DE.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df = pd.DataFrame({m: i["stats"] for m, i in data.items()}).T
            summary_df.index.name = "Method"
            summary_df.to_excel(writer, sheet_name="Summary")

            rows, index = [], []
            for method, info in data.items():
                for key, tag in (("mean", "Mean"), ("std", "Std"), ("best_curve", "Best")):
                    rows.append(info[key])
                    index.append(f"{method}_{tag}")
            width = max(len(r) for r in rows)
            padded = [np.pad(r, (0, width - len(r)), constant_values=np.nan) for r in rows]
            pd.DataFrame(padded, index=index,
                         columns=[f"Gen_{i + 1}" for i in range(width)]
                         ).to_excel(writer, sheet_name="Mean_Curves")

            finals = pd.DataFrame({m: pd.Series(i["finals"]) for m, i in data.items()})
            finals.index = [f"Run_{i + 1}" for i in range(len(finals))]
            finals.to_excel(writer, sheet_name="Final_Solutions")
        print(f"  -> {xlsx_path}")

        png_path = os.path.join(OUTPUT_DIR, f"{bit}bit_with_DE.png")
        plot_comparison(bit, data, png_path)
        print(f"  -> {png_path}")

        for method, info in data.items():
            overall.append({"bit": bit, "method": method, **info["stats"]})

    if de_data:
        de_png = os.path.join(OUTPUT_DIR, "DE_only_convergence.png")
        plot_de_only(de_data, de_png)
        print(f"\n[System] DE 收斂圖：{de_png}")

    if overall:
        df = pd.DataFrame(overall).set_index(["bit", "method"])
        out = os.path.join(OUTPUT_DIR, "overall_summary_with_DE.xlsx")
        df.to_excel(out)
        print(f"[System] 總表：{out}")
        print(df.to_string())


if __name__ == "__main__":
    main()
