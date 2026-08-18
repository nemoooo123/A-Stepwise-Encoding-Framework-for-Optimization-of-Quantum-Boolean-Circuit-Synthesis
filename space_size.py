"""
計算「編碼空間」的真實大小（可達解的數量），用來判斷窮舉是否可行。

關鍵觀察（決定了空間為什麼比 2^(bit 數) 小很多）：
  L1 priority_weights 只透過 sorted(..., reverse=True) 決定 cycle 的處理順序
     -> 只有相對順序有意義，共 C! 種
  L2 entry_points 經過 % node_mapping_index -> 每個 cycle 只有 limit_i 種
  L3 mid_node_matrix 同樣只決定 bit-flip 的順序 -> 每個 step d! 種
  L4 operation_sequences 經過 repair_sequence_logic 強制 0/1 個數相等
     -> 長度 2d-2 的平衡向量，共 C(2d-2, d-1) 種
  另外每個 cycle 只會用到 total_steps-1 個 step（entry 決定跳過哪一個），
  被跳過的 step 其 L3/L4 不影響結果。
"""
from math import comb, factorial

from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode


def cycle_profile(q3, q4, table, cyc_idx):
    """回傳 (limit, total_steps_adj, [每個 step 的 (d, L4 長度)])。"""
    steps = []
    for s in range(len(q3[cyc_idx])):
        d = len(q3[cyc_idx][s])          # sentinel 時為 1
        l4 = len(q4[cyc_idx][s])         # sentinel 時為 1
        steps.append((d, l4))
    total = len(q3[cyc_idx])
    return table[cyc_idx], (2 if total == 1 else total), steps


def step_choices(d, l4_len):
    """單一 step 的可達組合數：L3 的 d! 乘上 L4 的平衡向量數。"""
    if d == 1:                            # sentinel：L3=[999]，L4=[0]，各 1 種
        return 1
    return factorial(d) * comb(l4_len, l4_len // 2)


def used_steps(entry, total_steps_adj, n_real_steps):
    """synthesize_route 實際走訪的 step 索引（環狀，走 total_steps_adj-1 步）。"""
    out, ptr = [], entry
    for _ in range(total_steps_adj - 1):
        if ptr >= total_steps_adj:
            ptr = 0
        out.append(min(ptr, n_real_steps - 1))
        ptr += 1
    return out


def space_size(target):
    cycles, _ = find_cycles(target, check_zero_gate=True)
    n_cycles = len(cycles)
    _, _, q3, q4, table, traj = build_encode(cycles)

    per_cycle = []
    for i in range(n_cycles):
        limit, total_adj, steps = cycle_profile(q3, q4, table, i)
        # 對每個 entry，只算「實際會用到的 step」的組合數，再把 entry 加總
        total = 0
        for e in range(limit):
            prod = 1
            for s in used_steps(e, total_adj, len(steps)):
                prod *= step_choices(*steps[s])
            total += prod
        per_cycle.append({"len": len(traj[i]) - 1, "limit": limit,
                          "steps": steps, "combos": total})

    size = factorial(n_cycles)
    for c in per_cycle:
        size *= c["combos"]
    return n_cycles, per_cycle, size


if __name__ == "__main__":
    loader = DataLoader()
    for nbits in (3, 4):
        n_problems = len(loader.data_map[nbits])
        print(f"\n{'='*70}\n{nbits}-bit（{n_problems} 題）\n{'='*70}")
        for p in range(1, n_problems + 1):
            target = loader.get_output(nbits, p)
            n_cyc, per_cycle, size = space_size(target)
            lens = [c["len"] for c in per_cycle]
            print(f"  第 {p} 題: {n_cyc} 個 cycle，長度 {lens}，"
                  f"空間大小 = {size:,}  (~1e{len(str(size))-1})")
