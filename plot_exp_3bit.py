"""
畫 exp/3_bit 的收斂圖（AE-QTS vs DE）。

exp/3_bit 底下有 5 組 3-bit 目標電路（_1 ~ _5，各自最佳解不同），
每個 xlsx 已經是 100 次 trial x 1000 代，所以不需要合併，直接讀 Trial_* 統計。

輸出：exp_3bit_plots/
  3bit_case{i}_convergence.png  單一組的 AE-QTS vs DE 收斂圖（全程 + 前 200 代放大）
  3bit_all_cases.png            5 組總覽（2x3 子圖）
  3bit_mode_count.png           Mode_Count 收斂總覽
  3bit_entropy.png              AE-QTS 的 Entropy_Q1~Q4 收斂總覽
  3bit_summary.xlsx             各組各方法的統計摘要與平均收斂曲線
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
EXP_ROOT = os.path.join("exp", "3_bit")
OUTPUT_DIR = "exp_3bit_plots"
BIT = 3
CASES = [1, 2, 3, 4, 5]
METHODS = ["AE-QTS", "DE"]
ZOOM_GENS = 200          # 前段放大的代數範圍
ENTROPY_SHEETS = ["Entropy_Q1", "Entropy_Q2", "Entropy_Q3", "Entropy_Q4"]

METHOD_STYLE = {
    "AE-QTS": {"color": "#F40E0E", "lw": 2.0, "ls": "-"},
    "DE":     {"color": "#FF7F0E", "lw": 1.8, "ls": "--"},
}


# ---------- 讀取 ----------

def load_trials(method, case, sheet="Total_Gate_Count"):
    """讀單一 xlsx 的 Trial_* 曲線，回傳 (curves, exec_times)；找不到回傳 (None, None)。"""
    path = os.path.join(EXP_ROOT, f"{method}_Results", f"{method}_{BIT}_{case}.xlsx")
    if not os.path.exists(path):
        return None, None

    df = pd.read_excel(path, sheet_name=sheet, index_col=0)
    df.index = df.index.astype(str).str.strip()
    trials = df[df.index.str.startswith("Trial_")]

    gen_cols = [c for c in trials.columns if str(c).startswith("Gen_")]
    curves = trials[gen_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    times = None
    if "Execution_Time(s)" in trials.columns:
        times = pd.to_numeric(trials["Execution_Time(s)"], errors="coerce").to_numpy(dtype=float)
    return curves, times


def summarize(curves, times=None):
    finals = curves[:, -1]
    mean = curves.mean(axis=0)
    # 平均曲線第一次抵達最終值的代數 = 收斂代數
    conv_gen = int(np.argmax(mean <= mean[-1] + 1e-9)) + 1
    return {
        "mean": mean,
        "std": curves.std(axis=0, ddof=0),
        "best_curve": curves.min(axis=0),
        "finals": finals,
        "stats": {
            "runs": len(curves),
            "generations": curves.shape[1],
            "mean_best": float(finals.mean()),
            "std_best": float(finals.std(ddof=0)),
            "best": float(finals.min()),
            "worst": float(finals.max()),
            "best_hit": int((finals == finals.min()).sum()),
            "median": float(np.median(finals)),
            "converge_gen": conv_gen,
            "mean_time_s": None if times is None else float(np.nanmean(times)),
        },
    }


def load_case(case, sheet="Total_Gate_Count"):
    data = {}
    for method in METHODS:
        curves, times = load_trials(method, case, sheet)
        if curves is None:
            print(f"  [warn] 找不到 {method}_{BIT}_{case}.xlsx（{sheet}）")
            continue
        data[method] = summarize(curves, times)
    return data


# ---------- 繪圖 ----------

def draw_curves(ax, data, xmax=None, show_best=True, annotate=True, ylabel="Gate Count"):
    """把各方法的平均±std（與最佳曲線）畫在同一個 ax 上。"""
    for method, info in data.items():
        style = METHOD_STYLE[method]
        mean, std = info["mean"], info["std"]
        n = len(mean) if xmax is None else min(xmax, len(mean))
        gens = np.arange(1, n + 1)
        s = info["stats"]

        label = (f"{method}  (mean {s['mean_best']:.2f}±{s['std_best']:.2f}, "
                 f"best {s['best']:.0f}×{s['best_hit']})")
        ax.plot(gens, mean[:n], color=style["color"], lw=style["lw"], ls=style["ls"], label=label)
        ax.fill_between(gens, (mean - std)[:n], (mean + std)[:n], color=style["color"], alpha=0.13)
        if show_best:
            ax.plot(gens, info["best_curve"][:n], color=style["color"], lw=1.0,
                    ls=(0, (1, 2)), alpha=0.7)
        ax.axhline(s["mean_best"], color=style["color"], lw=0.9, ls=":", alpha=0.7)

    if annotate:
        last = max(len(i["mean"]) if xmax is None else min(xmax, len(i["mean"]))
                   for i in data.values())
        for k, (method, info) in enumerate(
                sorted(data.items(), key=lambda kv: kv[1]["stats"]["mean_best"])):
            s = info["stats"]
            color = METHOD_STYLE[method]["color"]
            ax.annotate(f"{method}: mean {s['mean_best']:.2f} / best {s['best']:.0f}",
                        xy=(last, s["mean_best"]), xytext=(-10, 12 + 16 * k),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=9, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, alpha=0.85))

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_case(case, data, save_path):
    """單一組：左邊全程、右邊前 ZOOM_GENS 代放大。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    n_gen = max(len(i["mean"]) for i in data.values())
    n_runs = max(i["stats"]["runs"] for i in data.values())

    draw_curves(ax1, data)
    ax1.set_title(f"Full run (1–{n_gen} generations)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, shadow=True, prop={"size": 9})

    draw_curves(ax2, data, xmax=ZOOM_GENS, annotate=False)
    ax2.set_title(f"Zoom: first {ZOOM_GENS} generations", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, shadow=True, prop={"size": 9})

    fig.suptitle(f"{BIT}-bit Case {case} Convergence — AE-QTS vs DE "
                 f"({n_runs} runs each)  |  dotted = best of {n_runs} runs",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_overview(all_data, save_path, title, ylabel="Gate Count", xmax=None):
    """5 組總覽：2x3 子圖。"""
    cases = sorted(all_data)
    ncol = 3
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.0 * nrow), squeeze=False)

    for ax, case in zip(axes.ravel(), cases):
        data = all_data[case]
        draw_curves(ax, data, xmax=xmax, show_best=False, annotate=False, ylabel=ylabel)
        bits = " | ".join(f"{m} {i['stats']['mean_best']:.2f}" for m, i in data.items())
        ax.set_title(f"Case {case}  ({bits})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_entropy(save_path):
    """AE-QTS 的 Entropy_Q1~Q4 平均收斂（每組一個子圖）。"""
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]
    ncol = 3
    nrow = int(np.ceil(len(CASES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.0 * nrow), squeeze=False)

    plotted_any = False
    for ax, case in zip(axes.ravel(), CASES):
        for sheet, color in zip(ENTROPY_SHEETS, colors):
            curves, _ = load_trials("AE-QTS", case, sheet)
            if curves is None:
                continue
            mean = curves.mean(axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, color=color, lw=1.6, label=sheet)
            plotted_any = True
        ax.set_title(f"Case {case}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Entropy")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(CASES):]:
        ax.axis("off")

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(f"AE-QTS Entropy Convergence — {BIT}-bit (100 runs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


# ---------- 主流程 ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gate_data, mode_data, overall = {}, {}, []

    for case in CASES:
        print(f"\n=== {BIT}-bit case {case} ===")
        data = load_case(case)
        if not data:
            continue
        gate_data[case] = data
        for method, info in data.items():
            s = info["stats"]
            print(f"  {method:<8}: {s['runs']} runs x {s['generations']} gens, "
                  f"mean {s['mean_best']:.2f}±{s['std_best']:.2f}, "
                  f"best {s['best']:.0f} (hit {s['best_hit']}), "
                  f"converge@gen {s['converge_gen']}")
            overall.append({"case": case, "method": method,
                            **{k: v for k, v in s.items()}})

        png = os.path.join(OUTPUT_DIR, f"{BIT}bit_case{case}_convergence.png")
        plot_case(case, data, png)
        print(f"  -> {png}")

        md = load_case(case, sheet="Mode_Count")
        if md:
            mode_data[case] = md

    if not gate_data:
        print("[warn] exp/3_bit 沒有讀到任何資料")
        return

    overview = os.path.join(OUTPUT_DIR, f"{BIT}bit_all_cases.png")
    plot_overview(gate_data, overview,
                  f"{BIT}-bit Convergence — 5 target circuits, AE-QTS vs DE (100 runs each)")
    print(f"\n[System] 總覽圖：{overview}")

    if mode_data:
        mode_png = os.path.join(OUTPUT_DIR, f"{BIT}bit_mode_count.png")
        plot_overview(mode_data, mode_png,
                      f"{BIT}-bit Mode Count Convergence — AE-QTS vs DE (100 runs each)",
                      ylabel="Mode Count")
        print(f"[System] Mode Count 圖：{mode_png}")

    ent_png = os.path.join(OUTPUT_DIR, f"{BIT}bit_entropy.png")
    if plot_entropy(ent_png):
        print(f"[System] Entropy 圖：{ent_png}")

    # --- 摘要 xlsx ---
    xlsx = os.path.join(OUTPUT_DIR, f"{BIT}bit_summary.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(overall).set_index(["case", "method"]).to_excel(writer, sheet_name="Summary")

        rows, index = [], []
        for case, data in gate_data.items():
            for method, info in data.items():
                for key, tag in (("mean", "Mean"), ("std", "Std"), ("best_curve", "Best")):
                    rows.append(info[key])
                    index.append(f"case{case}_{method}_{tag}")
        width = max(len(r) for r in rows)
        padded = [np.pad(r, (0, width - len(r)), constant_values=np.nan) for r in rows]
        pd.DataFrame(padded, index=index,
                     columns=[f"Gen_{i + 1}" for i in range(width)]
                     ).to_excel(writer, sheet_name="Mean_Curves")

        finals = {}
        for case, data in gate_data.items():
            for method, info in data.items():
                finals[f"case{case}_{method}"] = pd.Series(info["finals"])
        fdf = pd.DataFrame(finals)
        fdf.index = [f"Run_{i + 1}" for i in range(len(fdf))]
        fdf.to_excel(writer, sheet_name="Final_Solutions")
    print(f"[System] 摘要：{xlsx}")

    print()
    print(pd.DataFrame(overall).set_index(["case", "method"]).to_string())


if __name__ == "__main__":
    main()
