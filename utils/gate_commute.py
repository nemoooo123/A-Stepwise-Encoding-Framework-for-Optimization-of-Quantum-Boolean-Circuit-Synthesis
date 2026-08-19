"""
交換律化簡（commutation-aware simplification）。

原本 assemble_reversible_circuit 只消去「緊鄰的相同閘」。這裡多加一道：
如果兩個相同的閘中間夾的所有閘都能跟它交換位置，就把它移過去相鄰後一起消掉。

閘的表示（跟 topology.py 一致）
--------------------------------
長度 num_bits 的 list，0 / 1 = 控制位元（含極性），3 = target。
本框架產生的閘每個都恰好有一個 3，其餘 n-1 位都是控制位 ——
也就是說每個閘只交換兩個狀態：控制樣式配上 target = 0 與 target = 1。
換句話說，**一個閘 = 超立方體上的一條邊**。

交換律
------
兩個對換（transposition）可交換 <=> 兩者不相交，或完全相同。
以邊來看就是「兩條邊沒有共用頂點」，正是你說的「走法不相鄰就能互換」。

用閘的欄位表達，設 g1 = (t1, c1)、g2 = (t2, c2)：

  * t1 == t2（兩條平行邊）
        -> 一定可交換。相同則直接相消；不同則兩條平行邊不可能共用頂點。
           （即使控制條件同時成立，兩個閘都只翻同一個位元，先後無差別。）

  * t1 != t2
        -> 共用頂點存在 <=> c1 與 c2 在「除了 t1 和 t2 以外」的每個位置都相同
           （t1 那位由 c2 決定、t2 那位由 c1 決定，其餘必須一致）。
           所以：**只要在 t1、t2 以外有任一位不同，就可以交換。**

注意這比常見的 MPMCT 判準（「彼此的 target 不在對方的控制集合裡」）寬鬆很多。
本框架的閘是全控制的，t1 一定在 g2 的控制集合裡，用常見判準會得到「永遠不能
交換」的結論，白白錯失化簡機會。這裡用的是「支撐集是否相交」的精確條件。
"""

TARGET = 3


def gate_target(gate):
    """回傳 target 的位置索引；沒有恰好一個 3 就回傳 None。"""
    pos = -1
    for i, v in enumerate(gate):
        if v == TARGET:
            if pos != -1:
                return None            # 多個 target，走保守路線
            pos = i
    return None if pos == -1 else pos


def gates_commute(g1, g2):
    """兩個閘能不能互換位置。無法判定時回傳 False（保守）。"""
    t1 = gate_target(g1)
    t2 = gate_target(g2)
    if t1 is None or t2 is None:
        return False

    if t1 == t2:
        return True

    # t1 != t2：只要在 t1、t2 以外有一位控制條件互相矛盾，兩條邊就不相交
    for i in range(len(g1)):
        if i == t1 or i == t2:
            continue
        if g1[i] != g2[i]:
            return True
    return False


def simplify_commuting(gates):
    """
    反覆尋找「可以搬到一起」的相同閘對並消去，直到沒有為止。
    回傳新的 list，不改動輸入。
    """
    out = list(gates)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            gi = out[i]
            for j in range(i + 1, len(out)):
                if out[j] == gi:
                    del out[j]
                    del out[i]
                    changed = True
                    break
                if not gates_commute(gi, out[j]):
                    break              # 被卡住了，再往後也搬不過去
            if changed:
                break
    return out
