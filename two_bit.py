data='''
0000 1000
0001 1110
0010 0000
0011 0011
0100 0010
0101 0101
0110 1010
0111 0111
1000 0100
1001 1001
1010 1100
1011 1011
1100 0001
1101 1101
1110 0110
1111 1111
'''

# 自動解析轉換
output_list = []
for line in data.strip().split('\n'):
    parts = line.split()
    if len(parts) == 2:
        out_bin = parts[1]  # 取得右邊的輸出位元
        output_list.append(int(out_bin, 2))  # 轉成十進位並存入列表

# 印出結果供複製
print("轉換後的十進位列表：")
print(output_list)