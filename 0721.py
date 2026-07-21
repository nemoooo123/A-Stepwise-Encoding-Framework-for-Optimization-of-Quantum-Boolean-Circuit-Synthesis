def process_cancellation(data):
    # 為了不破壞原始數據，先拷貝一份（同時記錄每個陣列的初始長度）
    # 結構：{'list': [...], 'orig_len': int}
    chunks = [{'list': list(sub), 'orig_len': len(sub)} for sub in data]

    def try_cascade(curr, nxt):
        # 邊界互消：curr 尾端 與 nxt 開頭相同，且往內一格 (curr[-2] 與 nxt[1]) 也相同
        # 才允許消掉這一對（確保是「對稱」的互消，而不是隨便碰到相同值就消）
        if curr and nxt and curr[-1] == nxt[0]:
            if len(curr) >= 2 and len(nxt) >= 2 and curr[-2] == nxt[1]:
                curr.pop()
                nxt.pop(0)
                return True
        return False

    def run_cascades():
        # 對所有相鄰陣列反覆嘗試互消，直到沒有任何變化為止
        changed_any = False
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(chunks) - 1:
                curr = chunks[i]['list']
                nxt = chunks[i + 1]['list']
                if try_cascade(curr, nxt):
                    changed = True
                    changed_any = True
                else:
                    i += 1
        return changed_any

    def try_bridge_delete():
        # 特例：原始長度為 2 的陣列可以「跨著消」
        # 若它已被消到剩 0 個元素，或剩下最後 1 個元素但這個殘留值
        # 剛好與左右任一邊界相同，代表它只是個中介橋樑，可以整個刪除，
        # 讓左右兩個陣列直接相鄰，繼續互消
        for i in range(1, len(chunks) - 1):
            chunk = chunks[i]
            if chunk['orig_len'] != 2:
                continue
            lst = chunk['list']
            if len(lst) == 0:
                chunks.pop(i)
                return True
            if len(lst) == 1:
                left = chunks[i - 1]['list']
                right = chunks[i + 1]['list']
                residual = lst[0]
                if (left and left[-1] == residual) or (right and right[0] == residual):
                    chunks.pop(i)
                    return True
        return False

    changed = True
    while changed:
        changed = run_cascades()
        if try_bridge_delete():
            changed = True

    # 整理輸出結果
    result = [item['list'] for item in chunks]
    return result

# 測試資料
raw_data = [
    # [0, 8, 9, 13], [13, 9, 8], [8, 12], [12, 13, 5],
    # [5, 13, 9, 11, 10], [10, 11, 15], [15, 7, 3],
    # [3, 7, 6, 14], [14, 10, 11], [9, 1, 0, 2],
    # [2, 6], [6, 2, 0, 1], [1, 5, 4], [4, 5, 7]
    [0, 8, 12, 13], [13, 12, 8], [8, 12], [12, 13, 5], [5, 13, 15, 14, 10], [10, 14, 15], 
[15, 7, 3], [3, 11, 15, 14], [14, 15, 11], [9, 8, 0, 2], [2, 6], [6, 2, 0, 1], [1, 5, 4], [4, 5, 7]


]

output = process_cancellation(raw_data)

print("最終抵銷結果：")
for chunk in output:
    print(chunk)
