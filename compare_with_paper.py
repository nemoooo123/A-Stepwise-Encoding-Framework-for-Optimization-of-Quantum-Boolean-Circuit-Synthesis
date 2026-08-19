"""
把窮舉結果的閘數分布跟論文 IEEEQCE(1).pdf Fig. 3 的兩列對照。

Fig. 3 的數字是從 PDF 文字層抽出來的（不是看圖判讀）：
  Optimal  來自參考文獻 [27]-[29]，理論最佳
  Tsai's   Tsai & Kuo 方法解集合的窮舉結果
兩列都合計 40,320，可自行用 --check 驗證。

用法
----
  python compare_with_paper.py Exhaustive_3bit_all/all_40320.csv
  python compare_with_paper.py A/all_40320.csv B/all_40320.csv --labels 原本 加交換律
"""
import argparse
import csv
from collections import Counter

# --- 論文 Fig. 3（自 PDF 文字層擷取）---
PAPER = {
    "Optimal": {0: 1, 1: 12, 2: 90, 3: 476, 4: 1903, 5: 5472, 6: 10388,
                7: 11756, 8: 7347, 9: 2408, 10: 430, 11: 36, 12: 1},
    "Tsai's": {0: 1, 1: 12, 2: 90, 3: 476, 4: 1903, 5: 5376, 6: 9948,
               7: 10940, 8: 7431, 9: 3224, 10: 786, 11: 132, 12: 1},
}


def load(path):
    """讀 all_40320.csv，回傳 {閘數: 題數}。欄位可以是 best_gates 或 best。"""
    dist = Counter()
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in ("best_gates", "best"):
                if key in row and row[key] not in (None, ""):
                    dist[int(row[key])] += 1
                    break
    return dist


def stats(dist):
    n = sum(dist.values())
    s = sum(g * c for g, c in dist.items())
    return n, s, (s / n if n else 0), (max(dist) if dist else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="一個或多個 all_40320.csv")
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()

    cols = dict(PAPER)
    labels = args.labels or [f"檔案{i + 1}" for i in range(len(args.csvs))]
    for path, label in zip(args.csvs, labels):
        cols[label] = load(path)

    names = list(cols)
    gmax = max(max(d) for d in cols.values() if d)

    print("\n題數 / 平均 / 最大閘數")
    for nm in names:
        n, s, mean, mx = stats(cols[nm])
        warn = "" if n == 40320 else f"   <-- 不是 40,320！"
        print(f"  {nm:<12} 題數={n:>7,}  總閘數={s:>9,}  平均={mean:.4f}  最大={mx}{warn}")

    print("\n逐閘數題目數")
    head = "".join(f"{nm:>12}" for nm in names)
    print(f"  {'閘':>3}{head}")
    for g in range(gmax + 1):
        line = "".join(f"{cols[nm].get(g, 0):>12,}" for nm in names)
        print(f"  {g:>3}{line}")

    print("\n累積覆蓋（<= g 閘的題目數）")
    print(f"  {'閘':>3}{head}")
    cum = {nm: 0 for nm in names}
    for g in range(gmax + 1):
        for nm in names:
            cum[nm] += cols[nm].get(g, 0)
        print(f"  {g:>3}" + "".join(f"{cum[nm]:>12,}" for nm in names))

    # 跟 Tsai's 的差距（正 = 你比 Tsai's 多覆蓋）
    base = PAPER["Tsai's"]
    mine = [nm for nm in names if nm not in PAPER]
    if mine:
        print("\n累積覆蓋相對 Tsai's 的差（負 = 還差幾題）")
        print(f"  {'閘':>3}" + "".join(f"{nm:>12}" for nm in mine))
        cb = 0
        cm = {nm: 0 for nm in mine}
        for g in range(gmax + 1):
            cb += base.get(g, 0)
            for nm in mine:
                cm[nm] += cols[nm].get(g, 0)
            print(f"  {g:>3}" + "".join(f"{cm[nm] - cb:>+12,}" for nm in mine))


if __name__ == "__main__":
    main()
