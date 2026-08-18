#!/usr/bin/env bash
#
# 窮舉 3-bit 全部 40,320 個排列的編碼空間最佳解，把 CPU 跑滿。
#
# 用法
#   bash run_exhaustive_3bit.sh [並行數] [每份工作量]
#     並行數      預設 = nproc（你的機器是 12）
#     每份工作量  預設 500 萬個解；越小切得越細、負載越平均，但排程開銷越大
#
#   LIMIT=24 bash run_exhaustive_3bit.sh 6 2000000   # 只跑前 24 份，用來試跑
#
# 中斷後直接重跑同一行指令即可續跑：已完成的分片會跳過。
# 注意：改了「每份工作量」等於重新切分，舊分片檔名對不上、會整批重跑。
#
# 輸出目錄 Exhaustive_3bit_all/
#   tasks.csv          984 個等價類代表元與各自的空間大小
#   jobs.txt           實際派給 CPU 的工作清單（task part nparts）
#   parts/*.csv        每份工作的結果
#   merged_984.csv     合併成每個等價類一列
#   all_40320.csv      展開回全部 40,320 個排列
#   distribution.csv   最佳閘數分佈與總和
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONIOENCODING=utf-8

JOBS="${1:-$(nproc)}"
PARTS_TARGET="${2:-5000000}"
PYTHON_BIN="${PYTHON:-python}"
export PYTHON_BIN

OUT="Exhaustive_3bit_all"
PARTS="$OUT/parts"
mkdir -p "$PARTS"
export PARTS

echo "=== 3-bit 全題目窮舉 ==="
echo "    並行數 $JOBS，每份約 $PARTS_TARGET 個解"

echo "[1/6] 產生等價類清單（8! = 40,320 -> 984 個代表元）"
"$PYTHON_BIN" exhaustive_all.py --list-tasks > "$OUT/tasks.csv"
echo "      $(( $(wc -l < "$OUT/tasks.csv") - 1 )) 個代表元"

echo "[2/6] 切分工作"
"$PYTHON_BIN" exhaustive_all.py --plan --parts-target "$PARTS_TARGET" > "$OUT/jobs.txt"
if [ -n "${LIMIT:-}" ]; then
  head -n "$LIMIT" "$OUT/jobs.txt" > "$OUT/jobs.tmp"
  mv "$OUT/jobs.tmp" "$OUT/jobs.txt"
  echo "      [LIMIT] 只保留前 $LIMIT 份"
fi
echo "      共 $(wc -l < "$OUT/jobs.txt") 份工作"

# 單份工作：已有結果就跳過；先寫 .tmp 再 mv，中斷不會留下半截檔案
run_one() {
  local task part nparts f
  read -r task part nparts <<< "$1"
  f="$PARTS/p_${task}_${part}_${nparts}.csv"
  if [ -s "$f" ]; then
    return 0
  fi
  "$PYTHON_BIN" exhaustive_all.py --task "$task" --part "$part" \
      --nparts "$nparts" --out "$f.tmp"
  mv "$f.tmp" "$f"
}
export -f run_one

echo "[3/6] 開始窮舉（$JOBS 個並行，大的先跑）"
START=$(date +%s)
< "$OUT/jobs.txt" xargs -P "$JOBS" -I{} bash -c 'run_one "{}"'
echo "      耗時 $(( $(date +%s) - START )) 秒"

echo "[4/6] 合併分片"
"$PYTHON_BIN" exhaustive_all.py --merge "$PARTS" --out "$OUT/merged_984.csv"

echo "[5/6] 展開回 40,320 題"
"$PYTHON_BIN" exhaustive_all.py --expand "$OUT/merged_984.csv" --out "$OUT/all_40320.csv"

echo "[6/6] 統計 40,320 題的最佳解分佈與總和"
"$PYTHON_BIN" exhaustive_all.py --summary "$OUT/all_40320.csv" --out "$OUT/distribution.csv"

echo
echo "結果在 $OUT/"
