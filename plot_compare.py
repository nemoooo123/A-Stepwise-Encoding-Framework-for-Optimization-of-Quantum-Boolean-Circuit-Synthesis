"""
Compare the averaged convergence of two algorithms on the same problem.

Reads the per-trial convergence matrices that main.py exports to
    exp/{n}_bit/{ALGO}_Results/{ALGO}_{n}_{problem}.xlsx
averages every Gen_* column across trials, and plots one curve per algorithm.

Anything missing (no workbook, or only one of the two algorithms) is reported and
skipped, so a sweep never stops halfway.

Usage
    python plot_compare.py                     # every problem both algorithms have
    python plot_compare.py 6 1                 # just 6-bit problem 1
    python plot_compare.py 6                   # every 6-bit problem
    python plot_compare.py 3-6 1-5             # bits 3..6 x problems 1..5
    python plot_compare.py 3,6 all             # bits 3 and 6, every problem
    python plot_compare.py 6 1 --metric Unique_Gate_Count
    python plot_compare.py 6 1 --algos AE-QTS DE --no-band
"""
# python plot_compare.py 3-6 1-5
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Palette -----------------------------------------------------------------
# Categorical slots 1-3 of the reference data-viz palette, in fixed order:
# assigned per algorithm (never cycled, never re-assigned when the set changes).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"

FILENAME_RE = re.compile(r"^(?P<algo>.+)_(?P<bits>\d+)_(?P<problem>\d+)\.xlsx$")


def parse_spec(spec):
    """
    Parse a selector like "6", "3-6", "1,3,5", "3-4,6" or "all"/"*" into a sorted
    list of ints. None (or "all") means "whatever is on disk".
    """
    if spec is None or spec.lower() in ("all", "*"):
        return None
    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            try:
                lo, hi = int(lo), int(hi)
            except ValueError:
                raise SystemExit(f"[Error] cannot parse range '{part}' in '{spec}'")
            if lo > hi:
                lo, hi = hi, lo
            values.update(range(lo, hi + 1))
        else:
            try:
                values.add(int(part))
            except ValueError:
                raise SystemExit(f"[Error] cannot parse '{part}' in '{spec}'")
    if not values:
        raise SystemExit(f"[Error] empty selector: '{spec}'")
    return sorted(values)


def select_keys(available, bits_spec, problem_spec, algos):
    """
    Decide which (bits, problem) pairs to plot.

    When both selectors are explicit, the caller asked for a specific grid, so
    every requested combination is returned (missing ones get reported as skips
    by the caller). Otherwise only what exists on disk is considered.
    """
    if bits_spec is not None and problem_spec is not None:
        return [(b, q) for b in bits_spec for q in problem_spec]

    keys = sorted(
        k for k, v in available.items()
        if (bits_spec is None or k[0] in bits_spec)
        and (problem_spec is None or k[1] in problem_spec)
        and all(a in v for a in algos)
    )
    return keys


def discover(root):
    """Map (bits, problem) -> {algo: path} for every result workbook under root."""
    found = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("~$"):  # Excel lock file
                continue
            m = FILENAME_RE.match(name)
            if not m:
                continue
            key = (int(m.group("bits")), int(m.group("problem")))
            found.setdefault(key, {})[m.group("algo")] = os.path.join(dirpath, name)
    return found


def load_curve(path, metric):
    """
    Return (mean, std, n_trials, best_final, exec_times) over the Gen_* columns
    of `metric`.

    Averages the Trial_* rows directly rather than trusting the stored
    Average_Convergence row, and cross-checks the two.
    """
    sheets = pd.ExcelFile(path).sheet_names
    if metric not in sheets:
        # ValueError, not SystemExit: a sweep should skip this one and carry on.
        raise ValueError(
            f"'{metric}' sheet not in {os.path.basename(path)}. Available: {sheets}"
        )

    df = pd.read_excel(path, sheet_name=metric, index_col=0)
    gen_cols = [c for c in df.columns if str(c).startswith("Gen_")]
    gen_cols.sort(key=lambda c: int(str(c).split("_")[1]))

    trials = df.loc[[i for i in df.index if str(i).startswith("Trial_")], gen_cols]
    values = trials.to_numpy(dtype=float)

    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)

    # Sanity check against the row main.py already wrote.
    if "Average_Convergence" in df.index:
        stored = df.loc["Average_Convergence", gen_cols].to_numpy(dtype=float)
        gap = np.nanmax(np.abs(stored - mean))
        if gap > 1e-6:
            print(f"    [Warn] recomputed mean differs from stored row by {gap:.3g}")

    exec_times = None
    if "Execution_Time(s)" in df.columns:
        et = df.loc[trials.index, "Execution_Time(s)"].to_numpy(dtype=float)
        if np.isfinite(et).any():
            exec_times = et

    # Best single trial = lowest final-generation value across trials.
    best_final = float(np.nanmin(values[:, -1]))

    return mean, std, values.shape[0], best_final, exec_times


def plot(bits, problem, curves, metric, out_path, show_band):
    """One line per algorithm; x = generation, y = mean best gate count."""
    n_gen = min(len(c["mean"]) for c in curves.values())
    x = np.arange(1, n_gen + 1)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ends = []
    for slot, (algo, c) in enumerate(curves.items()):
        color = SERIES_COLORS[slot % len(SERIES_COLORS)]
        mean = c["mean"][:n_gen]
        if show_band:
            std = c["std"][:n_gen]
            ax.fill_between(x, mean - std, mean + std, color=color,
                            alpha=0.13, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.0, label=algo,
                solid_capstyle="round", zorder=3)
        ends.append((mean[-1], algo))

    # Direct labels at the line ends, so identity is never colour-alone. Final
    # values are often within a gate or two of each other, so push the labels
    # apart vertically until they stop overlapping.
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.05
    label_y = None
    for value, algo in sorted(ends, reverse=True):
        label_y = value if label_y is None else min(value, label_y - min_gap)
        ax.annotate(f"{algo}  {value:.1f}", xy=(n_gen, label_y),
                    xytext=(8, 0), textcoords="offset points",
                    annotation_clip=False, color=INK_SECONDARY,
                    fontsize=9, va="center", zorder=4)

    # Recessive grid and axes; no top/right spines.
    ax.grid(axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)

    pretty = metric.replace("_", " ")
    ax.set_xlabel("Generation", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(f"Mean {pretty.lower()}", color=INK_SECONDARY, fontsize=10)
    ax.set_xlim(1, n_gen)

    n_trials = {c["n_trials"] for c in curves.values()}
    trial_note = f"{n_trials.pop()} trials" if len(n_trials) == 1 else "trials differ"
    ax.set_title(f"{pretty} convergence  |  {bits}-bit problem {problem}",
                 color=INK_PRIMARY, fontsize=13, pad=30, loc="left")
    ax.text(0, 1.015, f"mean of {trial_note} per generation"
                      + (", shaded band = ±1 std" if show_band else ""),
            transform=ax.transAxes, color=INK_MUTED, fontsize=9, va="bottom")

    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # Headroom on the right for the direct labels, which sit outside the axes.
    fig.subplots_adjust(right=0.83)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bits", nargs="?",
                   help="bit width: 6, 3-6, 3,6 or all (default: all)")
    p.add_argument("problem", nargs="?",
                   help="problem index: 1, 1-5, 1,3,5 or all (default: all)")
    p.add_argument("--root", default="exp", help="results directory (default: exp)")
    p.add_argument("--metric", default="Total_Gate_Count",
                   help="sheet name to plot (default: Total_Gate_Count)")
    p.add_argument("--algos", nargs="+", default=["AE-QTS", "DE"],
                   help="algorithms to compare, in colour order (default: AE-QTS DE)")
    p.add_argument("--outdir", default=None,
                   help="where to write the PNGs (default: <root>/comparison)")
    p.add_argument("--no-band", action="store_true", help="hide the ±1 std band")
    p.add_argument("--csv", action="store_true",
                   help="also write the averaged curves as CSV")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"[Error] results directory not found: {args.root}")

    available = discover(args.root)
    if not available:
        raise SystemExit(f"[Error] no *_<bits>_<problem>.xlsx found under {args.root}")

    keys = select_keys(available, parse_spec(args.bits), parse_spec(args.problem),
                       args.algos)
    if not keys:
        raise SystemExit(
            f"[Error] nothing to plot under {args.root} for bits={args.bits or 'all'}, "
            f"problem={args.problem or 'all'}, algos={args.algos}.\n"
            f"        found: "
            + ", ".join(f"{b}_{q}={sorted(v)}" for (b, q), v in sorted(available.items()))
        )
    print(f"[System] {len(keys)} problem(s) requested: "
          + ", ".join(f"{b}_{q}" for b, q in keys))

    outdir = args.outdir or os.path.join(args.root, "comparison")
    os.makedirs(outdir, exist_ok=True)

    done, skipped = [], []
    for bits, problem in keys:
        paths = available.get((bits, problem), {})
        missing = [a for a in args.algos if a not in paths]
        if missing:
            reason = "no workbook" if not paths else f"missing {missing}"
            print(f"\n[Skip] {bits}-bit problem {problem}: {reason}"
                  + (f" (has {sorted(paths)})" if paths else ""))
            skipped.append(f"{bits}_{problem}")
            continue

        print(f"\n=== {bits}-bit problem {problem} | {args.metric} ===")
        try:
            curves = {}
            for algo in args.algos:
                mean, std, n_trials, best_final, exec_times = load_curve(
                    paths[algo], args.metric)
                curves[algo] = {"mean": mean, "std": std, "n_trials": n_trials}
                note = ""
                if exec_times is not None:
                    note = f" | avg {np.nanmean(exec_times):7.2f}s/trial"
                print(f"  {algo:8s} {n_trials:3d} trials, {len(mean):4d} gens"
                      f" | gen1 {mean[0]:7.2f} -> final {mean[-1]:7.2f}"
                      f" +/- {std[-1]:5.2f} | best trial {best_final:7.2f}{note}")

            base = f"{'_vs_'.join(args.algos)}_{bits}_{problem}_{args.metric}"
            png = os.path.join(outdir, base + ".png")
            plot(bits, problem, curves, args.metric, png, show_band=not args.no_band)
            print(f"  [Saved] {png}")

            if args.csv:
                n_gen = min(len(c["mean"]) for c in curves.values())
                out = pd.DataFrame({"Generation": np.arange(1, n_gen + 1)})
                for algo, c in curves.items():
                    out[f"{algo}_mean"] = c["mean"][:n_gen]
                    out[f"{algo}_std"] = c["std"][:n_gen]
                csv = os.path.join(outdir, base + ".csv")
                out.to_csv(csv, index=False)
                print(f"  [Saved] {csv}")
        except Exception as exc:
            # One unreadable workbook must not abort the whole sweep.
            print(f"  [Skip] {exc}")
            skipped.append(f"{bits}_{problem}")
            continue

        done.append(f"{bits}_{problem}")

    print(f"\n[System] plotted {len(done)}: " + (", ".join(done) or "-"))
    if skipped:
        print(f"[System] skipped {len(skipped)}: " + ", ".join(skipped))


if __name__ == "__main__":
    sys.exit(main())
