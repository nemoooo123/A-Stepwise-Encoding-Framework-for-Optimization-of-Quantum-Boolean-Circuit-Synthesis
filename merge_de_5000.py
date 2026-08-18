"""
合併 exp_5000_1 ~ exp_5000_4 的 DE 結果（各 25 次 trial，合併後 100 次）。

沿用 merge_5000.py 的合併與繪圖流程，只是把來源資料夾與演算法換成 DE。

輸出：DE_5000_merged/{bit}_bit/DE_Results/
"""

import merge_5000 as m

m.SOURCE_FOLDERS = ["exp_5000_1", "exp_5000_2", "exp_5000_3", "exp_5000_4"]
m.OUTPUT_ROOT = "DE_5000_merged"
m.ALGO = "DE"
m.RESULT_DIR = "DE_Results"

if __name__ == "__main__":
    m.main()
