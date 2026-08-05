from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _clog2(v: int) -> int:
    """⌈log2(v)⌉，表示 0..v-1 需要幾個位元"""
    if v <= 1:
        return 0
    return (v - 1).bit_length()


def _popcount(x: int) -> int:
    """回傳整數 x 的 1 位元個數（Hamming weight）"""
    return int(x).bit_count()


def _diff_bits(u: int, v: int) -> List[int]:
    """回傳 u 與 v 相異位元的索引集合（由小到大）"""
    d = u ^ v
    bits = []
    while d:
        b = (d & -d).bit_length() - 1  # 最低位設定位元索引
        bits.append(b)
        d &= d - 1
    return bits


def _diff_bits_msb(u: int, v: int, n: int) -> List[int]:
    """回傳 u 與 v 相異位元的索引"""
    d = u ^ v
    out: List[int] = []
    for p in range(n):          # p = MSB-first 位置（0 = 最高位）
        b = n - 1 - p           # 對應的 LSB 位元索引
        if (d >> b) & 1:
            out.append(b)
    return out


def _comb_bit_length(ell: int) -> int:
    """
    單一換位位元長度
    b_comb = ⌈log2 C(2(ℓ−1), ℓ−1)⌉
    """
    if ell <= 1:
        return 0
    return _clog2(math.comb(2 * (ell - 1), ell - 1))


def _path_bit_length(ell: int) -> int:
    """單一邊的 path（L3）欄位位元長度 = ℓ × ⌈log2 ℓ⌉"""
    if ell <= 1:
        return 0
    return ell * _clog2(ell)

@dataclass(frozen=True)
class Gate:
    target_bit: int
    control_value: int


def gate_from_edge(u: int, v: int) -> Gate:
    """由超立方體邊 (u, v)建立對應的換位閘 target_bit = u⊕v, control_value = u"""
    d = u ^ v
    target_bit = d.bit_length() - 1
    control_value = u & ~(1 << target_bit)
    return Gate(target_bit, control_value)


def apply_gate(gate: Gate, state: int) -> int:
    """若 state 除 target 位元外的其餘位元 == control_value，則反轉target位元"""
    if (state & ~(1 << gate.target_bit)) == gate.control_value:
        return state ^ (1 << gate.target_bit)
    return state


def simulate(gates: Sequence[Gate], n: int) -> List[int]:
    """模擬序列，回傳其置換"""
    result = []
    for x in range(1 << n):
        s = x
        for g in gates:
            s = apply_gate(g, s)
        result.append(s)
    return result


def simplify(gates: Sequence[Gate]) -> List[Gate]:
    """相鄰兩個相同閘互消（UU = I）以化簡序列"""
    stack: List[Gate] = []
    for g in gates:
        if stack and stack[-1] == g:
            stack.pop()
        else:
            stack.append(g)
    return stack



def decompose_cycles(perm: Sequence[int]) -> List[List[int]]:
    """
    Step 1：將 perm 去除不動點後分解為循環

    由最小的「未訪問且非不動點」狀態起始，
    依 perm 追蹤，直到回到起點每個循環 C_i = 有序節點列表
    [a_0, a_1, ..., a_{m-1}]，其中 a_{t+1} = perm[a_t]

    ex：perm=[7,2,3,1,4,5,6,0] ⇒ [[0,7], [1,2,3]]
    恆等置換 ⇒ 回傳 []
    """
    n_states = len(perm)
    visited = [False] * n_states
    cycles: List[List[int]] = []
    for start in range(n_states):
        if visited[start] or perm[start] == start:
            # 已訪問或為不動點：跳過
            visited[start] = True
            continue
        cycle = []
        x = start
        while not visited[x]:
            visited[x] = True
            cycle.append(x)
            x = perm[x]
        cycles.append(cycle)
    return cycles

def _rungs_for_path(x: int, path: Sequence[int]) -> List[Gate]:
    """
    給定起點 x 與翻轉順序 path=[b_0,...,b_{ℓ-1}]
    [F_1,...,F_ℓ]，其中 F_t = gate_from_edge(v_{t-1}, v_t)，
    v_0=x, v_t = v_{t-1} ⊕ 2^{b_{t-1}}
    """
    rungs = []
    cur = x
    for b in path:
        nxt = cur ^ (1 << b)
        rungs.append(gate_from_edge(cur, nxt))
        cur = nxt
    return rungs


def _unrank_interleave(left: List[Gate], right: List[Gate], rank: int) -> List[Gate]:
    """
    共 C(|left|+|right|, |left|) 種交錯決定「下一個取 left 或 right
    若剩餘 left 有 l 個、right 有 r 個，則以 left 開頭的交錯數 =C(l+r−1, l−1)
    rank 小於此數 ⇒ 取 left，否則扣除後取 right
    """
    out: List[Gate] = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        lc = len(left) - i   # 剩餘 left 數
        rc = len(right) - j  # 剩餘 right 數
        c = math.comb(lc + rc - 1, lc - 1)  # 以 left 開頭的交錯數
        if rank < c:
            out.append(left[i])
            i += 1
        else:
            rank -= c
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def _ladder_variant(rungs: List[Gate], comb: int) -> List[Gate]:
    """
    Step 3-2：由 comb 值決定梯形變體

    對 rung 序列 [F_1..F_ℓ]：
      - 選 apex 索引 a ∈ {1..ℓ}（apex = F_a）
      - 左鏈 L = [F_1,...,F_{a-1}]、右鏈 R = [F_ℓ,...,F_{a+1}]
      - prefix = L 與 R 的任意交錯
      - suffix = reversed(L) 與 reversed(R)的任意交錯
      - 閘序列 = prefix + [F_a] + suffix，共 2ℓ−1 個閘

    L 的閘只作用於 x 側節點 {v_0..v_{a-1}}、R 的閘只作用於 y 側節點
    {v_a..v_ℓ}，支撐不相交故 L、R 的閘兩兩可交換 ⇒ 任何交錯下
    prefix ≡ P_L∘P_R、suffix ≡ P_L⁻¹∘P_R⁻¹ = prefix⁻¹⇒ 整體 = P⁻¹∘F_a∘P
    """
    ell = len(rungs)
    if ell == 1:
        return [rungs[0]]
    total = math.comb(2 * (ell - 1), ell - 1)
    v = comb % total
    # 依區塊大小 s_a = C(ℓ-1, a-1)² 找 apex 索引 a 與區塊內餘數
    a = 1
    while True:
        s = math.comb(ell - 1, a - 1) ** 2
        if v < s:
            break
        v -= s
        a += 1
    c = math.comb(ell - 1, a - 1)
    prefix_rank, suffix_rank = divmod(v, c)
    left = rungs[: a - 1]         # L = [F_1,...,F_{a-1}]
    right = rungs[a:][::-1]       # R = [F_ℓ,...,F_{a+1}]
    prefix = _unrank_interleave(left, right, prefix_rank)
    suffix = _unrank_interleave(left[::-1], right[::-1], suffix_rank)
    return prefix + [rungs[a - 1]] + suffix


def expand_transposition(x: int, y: int, comb: int, path: Sequence[int]) -> List[Gate]:
    """
    將換位 T(x, y) 依指定 path 與 comb 展開為閘序列
    ℓ=1：單一閘 gate_from_edge(x, y)
    ℓ≥2：先由 path 得 rung 序列，再由 comb 決定梯形變體
    """
    if x == y:
        return []
    ell = len(path)
    if ell == 1:
        return [gate_from_edge(x, y)]
    rungs = _rungs_for_path(x, path)
    return _ladder_variant(rungs, comb)


def _merge_apply(reduced: List[Gate], new_gates: Sequence[Gate]) -> None:
    for g in new_gates:
        if reduced and reduced[-1] == g:
            reduced.pop()
        else:
            reduced.append(g)


def _read_field(bits: Sequence[int], start: int, width: int) -> int:
    """自 bits[start:start+width] 以 big-endian 讀出整數；width=0 回傳 0"""
    val = 0
    for k in range(width):
        val = (val << 1) | int(bits[start + k])
    return val


def _decode_path(
    bits: Sequence[int], start: int, ell: int, k: int, idx_list: Sequence[int]
) -> List[int]:
    """Step 3-1（L3）：由基因組 path 欄位解出翻位順序"""
    prios = [_read_field(bits, start + t * k, k) for t in range(ell)]
    # sorted 為穩定排序 ⇒ 同優先權者維持 idx_list 順序
    order = sorted(range(ell), key=lambda t: -prios[t])
    return [idx_list[t] for t in order]


@dataclass
class _CycleLayout:
    """單一循環在基因組中的欄位佈局與靜態資訊"""
    index: int              # 循環原始索引 i
    nodes: List[int]        # 有序節點 [a_0,...,a_{m-1}]
    m: int                  # 循環長度（邊數）
    b_ord: int              # permv 欄位位元數
    b_edge: int             # cutedge 欄位位元數 
    edges: List[Tuple[int, int]]     # [(a_j, a_{(j+1)%m})]
    edge_ell: List[int]              # 各邊 Hamming 距離 ℓ
    path_bits: List[int]             # 各邊 path（L3）欄位位元數 = ℓ×⌈log2 ℓ⌉
    comb_bits: List[int]             # 各邊 comb 欄位位元數
    ord_pos: int            # permv 欄位在基因組中的起始位元位置
    edge_pos: int           # cutedge 欄位起始位置
    path_pos: List[int]     # 各邊 path 欄位起始位置
    comb_pos: List[int]     # 各邊 comb 欄位起始位置


class EncodingLayout:
    """Stepwise Encoding 的基因組佈局與解碼器"""

    def __init__(self, perm: Sequence[int]) -> None:
        self.perm: List[int] = list(perm)
        n_states = len(self.perm)
        self.n: int = (n_states - 1).bit_length() if n_states > 1 else 0

        cycles = decompose_cycles(self.perm)
        self.K: int = len(cycles)
        b_ord = _clog2(self.K)

        self._cycles: List[_CycleLayout] = []
        pos = 0
        for i, nodes in enumerate(cycles):
            m = len(nodes)
            edges = [(nodes[j], nodes[(j + 1) % m]) for j in range(m)]
            edge_ell = [_popcount(u ^ v) for (u, v) in edges]
            path_bits = [_path_bit_length(l) for l in edge_ell]
            comb_bits = [_comb_bit_length(l) for l in edge_ell]

            # Block_i = [permv | cutedge | P_i（各邊 path）| C_i（各邊 comb）]
            ord_pos = pos
            pos += b_ord
            edge_pos = pos
            b_edge = _clog2(m)
            pos += b_edge
            path_pos = []
            for pb in path_bits:
                path_pos.append(pos)
                pos += pb
            comb_pos = []
            for cb in comb_bits:
                comb_pos.append(pos)
                pos += cb

            self._cycles.append(
                _CycleLayout(
                    index=i,
                    nodes=nodes,
                    m=m,
                    b_ord=b_ord,
                    b_edge=b_edge,
                    edges=edges,
                    edge_ell=edge_ell,
                    path_bits=path_bits,
                    comb_bits=comb_bits,
                    ord_pos=ord_pos,
                    edge_pos=edge_pos,
                    path_pos=path_pos,
                    comb_pos=comb_pos,
                )
            )

        self.search_bit_length: int = pos

    @staticmethod
    def _read_field(bits: Sequence[int], start: int, width: int) -> int:
        """自 bits[start:start+width] 讀出整數"""
        return _read_field(bits, start, width)

    # 解碼

    def decode(self, bits: np.ndarray) -> List[Gate]:
        """
        將基因組 bits 解碼為閘序列
          1. 讀出各循環 permv、cutedge、各邊 path（L3）與 comb（L4）
          2. 循環處理順序 = sort key (−permv, index)
          3. 每循環切掉 cutedge 指定的邊，其餘 m−1 條邊沿環反序處理
          4. 每條邊依其 path 與 comb 展開，即時互消併入結果
        """
        if self.K == 0:
            return []

        # 欄位解碼
        decoded = []  # (cyc, permv, cutedge, [path per edge], [comb per edge])
        for cyc in self._cycles:
            if cyc.b_ord > 0:
                raw = _read_field(bits, cyc.ord_pos, cyc.b_ord)
                permv = raw % self.K
            else:
                permv = 0
            cut_raw = _read_field(bits, cyc.edge_pos, cyc.b_edge)
            cutedge = cut_raw % cyc.m

            paths: List[Optional[List[int]]] = []
            for j in range(cyc.m):
                ell = cyc.edge_ell[j]
                if ell <= 1:
                    paths.append(None)  # ℓ=1 無路徑自由度
                else:
                    u, v = cyc.edges[j]
                    idx_list = _diff_bits_msb(u, v, self.n)
                    paths.append(
                        _decode_path(
                            bits, cyc.path_pos[j], ell, _clog2(ell), idx_list
                        )
                    )
            combs = [
                _read_field(bits, cyc.comb_pos[j], cyc.comb_bits[j])
                for j in range(cyc.m)
            ]
            decoded.append((cyc, permv, cutedge, paths, combs))

        # 循環處理順序：permv 大者優先，permv 同值時 index 小者優先
        order = sorted(decoded, key=lambda t: (-t[1], t[0].index))

        reduced: List[Gate] = []
        for cyc, _permv, cutedge, paths, combs in order:
            m = cyc.m
            j = cutedge  # 被切邊索引（沿 Cedge）
            # 其餘 m−1 條邊沿環反序
            for step in range(m - 1):
                ei = (j - 1 - step) % m
                x, y = cyc.edges[ei]
                path = paths[ei]
                if path is None:
                    _merge_apply(reduced, [gate_from_edge(x, y)])
                else:
                    _merge_apply(
                        reduced, expand_transposition(x, y, combs[ei], path)
                    )

        return reduced


def paper_bit_length(perm: Sequence[int]) -> int:
    """
    依論文公式計算基因組總位元長度 BL

    對每個循環 C_i（長度 m_i、共 K 個循環）：
      b_ord,i        = ⌈log2 K⌉
      b_edge,i       = ⌈log2 m_i⌉
      totalb_path,i  = Σ_j Σ_{k=0..ℓ(e_ij)−2} ⌈log2 (ℓ(e_ij)−k)⌉
      totalb_comb,i  = Σ_j ⌈log2 C(2(ℓ−1), ℓ−1)⌉
      BL = Σ_i (b_ord,i + b_edge,i + totalb_path,i + totalb_comb,i)
    """
    cycles = decompose_cycles(perm)
    K = len(cycles)
    b_ord = _clog2(K)
    total = 0
    for nodes in cycles:
        m = len(nodes)
        b_edge = _clog2(m)
        edges = [(nodes[j], nodes[(j + 1) % m]) for j in range(m)]
        total_path = 0
        total_comb = 0
        for (u, v) in edges:
            ell = _popcount(u ^ v)
            for k in range(ell - 1):
                total_path += _clog2(ell - k)
            total_comb += _comb_bit_length(ell)
        total += b_ord + b_edge + total_path + total_comb
    return total


#基準函數
def hwb(n: int) -> List[int]:
    """Hidden Weighted Bit 基準置換"""
    if n <= 0:
        return [0]
    size = 1 << n
    mask = size - 1
    perm = []
    for x in range(size):
        r = _popcount(x) % n
        rotated = ((x << r) | (x >> (n - r))) & mask
        perm.append(rotated)
    return perm

#nbit自動生成
def random_permutation(n: int, seed: int) -> List[int]:
    """回傳 n-qubit（2^n 元素）的隨機置換，以固定 seed 保證可重現"""
    rng = np.random.default_rng(seed)
    return rng.permutation(1 << n).tolist()
