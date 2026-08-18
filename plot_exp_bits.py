"""
畫 exp_個位元 底下所有位元數的收斂圖（AE-QTS vs DE）。

exp_個位元/{n}_bit/{AE-QTS,DE}_Results/{method}_{n}_{case}.xlsx
每個 xlsx 已經是 100 次 trial x 1000 代，直接讀 Trial_* 統計，不需合併。
各位元的 case 數不同（3/5/6-bit 有 5 組，4-bit 3 組，7-bit 2 組，8/9/10-bit 1 組），
Entropy 的 qubit 數也不同，所以兩者都自動偵測。

輸出：exp_bits_plots/
  {n}bit_case{i}_convergence.png  單一組的 AE-QTS vs DE 收斂圖（全程 + 前 200 代放大）
  {n}bit_all_cases.png            該位元所有 case 的總覽
  {n}bit_mode_count.png           Mode_Count 收斂總覽
  {n}bit_entropy.png              AE-QTS 的 Entropy_Q* 收斂總覽
  {n}bit_summary.xlsx             該位元的統計摘要與平均收斂曲線
  all_bits_summary.xlsx           跨位元的總摘要
"""

import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
EXP_ROOT = "exp_個位元"
OUTPUT_DIR = "exp_bits_plots"
METHODS = ["AE-QTS", "DE"]
ZOOM_GENS = 200          # 前段放大的代數範圍

METHOD_STYLE = {
    "AE-QTS": {"color": "#F40E0E", "lw": 2.0, "ls": "-"},
    "DE":     {"color": "#FF7F0E", "lw": 1.8, "ls": "--"},
}


# ---------- 掃描資料夾 ----------

def discover():
    """回傳 {bit: [case, ...]}，只收 AE-QTS 與 DE 至少有一邊存在的 case。"""
    found = {}
    for name in sorted(os.listdir(EXP_ROOT)):
        m = re.fullmatch(r"(\d+)_bit", name)
        if not m or not os.path.isdir(os.path.join(EXP_ROOT, name)):
            continue
        bit = int(m.group(1))
        cases = set()
        for method in METHODS:
            d = os.path.join(EXP_ROOT, name, f"{method}_Results")
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                mm = re.fullmatch(rf"{re.escape(method)}_{bit}_(\d+)\.xlsx", f)
                if mm:
                    cases.add(int(mm.group(1)))
        if cases:
            found[bit] = sorted(cases)
    return dict(sorted(found.items()))


def xlsx_path(bit, method, case):
    return os.path.join(EXP_ROOT, f"{bit}_bit", f"{method}_Results",
                        f"{method}_{bit}_{case}.xlsx")


def entropy_sheets(bit, case):
    """AE-QTS 的 Entropy_Q* 分頁名稱（qubit 數隨位元數變化）。"""
    path = xlsx_path(bit, "AE-QTS", case)
    if not os.path.exists(path):
        return []
    names = pd.ExcelFile(path, engine="openpyxl").sheet_names
    ent = [s for s in names if re.fullmatch(r"Entropy_Q\d+", s)]
    return sorted(ent, key=lambda s: int(s.split("Q")[1]))


# ---------- 讀取 ----------

def load_trials(bit, method, case, sheet="Total_Gate_Count"):
    """讀單一 xlsx 的 Trial_* 曲線，回傳 (curves, exec_times)；找不到回傳 (None, None)。"""
    path = xlsx_path(bit, method, case)
    if not os.path.exists(path):
        return None, None

    try:
        df = pd.read_excel(path, sheet_name=sheet, index_col=0)
    except ValueError:      # 沒有這個分頁
        return None, None
    df.index = df.index.astype(str).str.strip()
    trials = df[df.index.str.startswith("Trial_")]
    if trials.empty:
        return None, None

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


def load_case(bit, case, sheet="Total_Gate_Count", warn=True):
    data = {}
    for method in METHODS:
        curves, times = load_trials(bit, method, case, sheet)
        if curves is None:
            if warn:
                print(f"  [warn] 找不到 {method}_{bit}_{case}.xlsx（{sheet}）")
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


def plot_case(bit, case, data, save_path):
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

    top = suptitle(fig, f"{bit}-bit Case {case} Convergence — AE-QTS vs DE "
                        f"({n_runs} runs each)  |  dotted = best of {n_runs} runs")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def grid(n, per_row=3, w=5.4, h=4.0, min_w=11.0):
    """子圖格線。min_w 保證畫布不會窄到讓 suptitle 被裁掉（case 數少時會遇到）。"""
    ncol = min(per_row, n)
    nrow = int(np.ceil(n / ncol))
    fig_w = max(w * ncol, min_w)
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, h * nrow), squeeze=False)
    return fig, axes.ravel()


def suptitle(fig, title, fontsize=14):
    """按畫布寬度折行後再下標題（14pt 粗體約每吋 9 個字）。"""
    per_line = max(20, int(fig.get_figwidth() * 9 * 14 / fontsize))
    text = "\n".join(textwrap.wrap(title, per_line)) or title
    fig.suptitle(text, fontsize=fontsize, fontweight="bold")
    # 標題行數越多，留給子圖的上緣就要壓低一點
    return 1 - 0.05 * text.count("\n") - 0.05


def plot_overview(all_data, save_path, title, ylabel="Gate Count", xmax=None):
    """同一位元數各 case 的總覽。"""
    cases = sorted(all_data)
    fig, axes = grid(len(cases))

    for ax, case in zip(axes, cases):
        data = all_data[case]
        draw_curves(ax, data, xmax=xmax, show_best=False, annotate=False, ylabel=ylabel)
        bits = " | ".join(f"{m} {i['stats']['mean_best']:.2f}" for m, i in data.items())
        ax.set_title(f"Case {case}  ({bits})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[len(cases):]:
        ax.axis("off")

    top = suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_entropy(bit, cases, save_path):
    """AE-QTS 的 Entropy_Q* 平均收斂（每組一個子圖）。"""
    cmap = plt.get_cmap("tab10")
    fig, axes = grid(len(cases))

    plotted_any = False
    for ax, case in zip(axes, cases):
        sheets = entropy_sheets(bit, case)
        for i, sheet in enumerate(sheets):
            curves, _ = load_trials(bit, "AE-QTS", case, sheet)
            if curves is None:
                continue
            mean = curves.mean(axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, color=cmap(i % 10), lw=1.4, label=sheet)
            plotted_any = True
        ax.set_title(f"Case {case}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Entropy")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=7, ncol=2)

    for ax in axes[len(cases):]:
        ax.axis("off")

    if not plotted_any:
        plt.close(fig)
        return False

    top = suptitle(fig, f"AE-QTS Entropy Convergence — {bit}-bit (100 runs)")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def write_summary(path, overall, gate_data_by_key):
    """overall: list of dict；gate_data_by_key: {label: info}"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(overall).set_index(["bit", "case", "method"]).to_excel(
            writer, sheet_name="Summary")

        rows, index = [], []
        for label, info in gate_data_by_key.items():
            for key, tag in (("mean", "Mean"), ("std", "Std"), ("best_curve", "Best")):
                rows.append(info[key])
                index.append(f"{label}_{tag}")
        if rows:
            width = max(len(r) for r in rows)
            padded = [np.pad(r, (0, width - len(r)), constant_values=np.nan) for r in rows]
            pd.DataFrame(padded, index=index,
                         columns=[f"Gen_{i + 1}" for i in range(width)]
                         ).to_excel(writer, sheet_name="Mean_Curves")

            finals = {label: pd.Series(info["finals"])
                      for label, info in gate_data_by_key.items()}
            fdf = pd.DataFrame(finals)
            fdf.index = [f"Run_{i + 1}" for i in range(len(fdf))]
            fdf.to_excel(writer, sheet_name="Final_Solutions")


# ---------- 主流程 ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    found = discover()
    if not found:
        print(f"[warn] {EXP_ROOT} 下沒有找到 *_bit 資料")
        return
    print(f"[System] 找到：" + "、".join(f"{b}-bit x{len(c)}" for b, c in found.items()))

    all_overall, all_curves = [], {}

    for bit, cases in found.items():
        gate_data, mode_data, overall = {}, {}, []

        for case in cases:
            print(f"\n=== {bit}-bit case {case} ===")
            data = load_case(bit, case)
            if not data:
                continue
            gate_data[case] = data
            for method, info in data.items():
                s = info["stats"]
                print(f"  {method:<8}: {s['runs']} runs x {s['generations']} gens, "
                      f"mean {s['mean_best']:.2f}±{s['std_best']:.2f}, "
                      f"best {s['best']:.0f} (hit {s['best_hit']}), "
                      f"converge@gen {s['converge_gen']}")
                row = {"bit": bit, "case": case, "method": method, **s}
                overall.append(row)
                all_overall.append(row)
                all_curves[f"{bit}bit_case{case}_{method}"] = info

            png = os.path.join(OUTPUT_DIR, f"{bit}bit_case{case}_convergence.png")
            plot_case(bit, case, data, png)
            print(f"  -> {png}")

            md = load_case(bit, case, sheet="Mode_Count", warn=False)
            if md:
                mode_data[case] = md

        if not gate_data:
            print(f"[warn] {bit}-bit 沒有讀到任何資料")
            continue

        overview = os.path.join(OUTPUT_DIR, f"{bit}bit_all_cases.png")
        plot_overview(gate_data, overview,
                      f"{bit}-bit Convergence — {len(gate_data)} target circuit(s), "
                      f"AE-QTS vs DE (100 runs each)")
        print(f"[System] {bit}-bit 總覽圖：{overview}")

        if mode_data:
            mode_png = os.path.join(OUTPUT_DIR, f"{bit}bit_mode_count.png")
            plot_overview(mode_data, mode_png,
                          f"{bit}-bit Mode Count Convergence — AE-QTS vs DE (100 runs each)",
                          ylabel="Mode Count")
            print(f"[System] {bit}-bit Mode Count 圖：{mode_png}")

        ent_png = os.path.join(OUTPUT_DIR, f"{bit}bit_entropy.png")
        if plot_entropy(bit, sorted(gate_data), ent_png):
            print(f"[System] {bit}-bit Entropy 圖：{ent_png}")

        bit_xlsx = os.path.join(OUTPUT_DIR, f"{bit}bit_summary.xlsx")
        write_summary(bit_xlsx,
                      [r for r in overall],
                      {f"{bit}bit_case{c}_{m}": i
                       for c, d in gate_data.items() for m, i in d.items()})
        print(f"[System] {bit}-bit 摘要：{bit_xlsx}")

    all_xlsx = os.path.join(OUTPUT_DIR, "all_bits_summary.xlsx")
    write_summary(all_xlsx, all_overall, all_curves)
    print(f"\n[System] 總摘要：{all_xlsx}")

    print()
    print(pd.DataFrame(all_overall).set_index(["bit", "case", "method"]).to_string())


if __name__ == "__main__":
    main()
