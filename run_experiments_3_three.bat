@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ==========================================
:: 1. 實驗參數設定
:: ==========================================
set PYTHON_FILE=main_three.py
set QUESTION_ID=3
set FOLDER=exp
set QUESTION_FOLDER=%~dp0question

:: --- 檢查 Python 檔案是否存在 ---
if not exist "%PYTHON_FILE%" (
    echo [錯誤] 找不到 %PYTHON_FILE%！
    pause
    exit
)

echo =======================================================
echo [啟動中心] 執行環境檢查通過
echo 目標檔案: %PYTHON_FILE%
echo 目標題目: tasks_part_%QUESTION_ID%.txt
echo 動作    : 同時啟動演算法 1 到 9
echo =======================================================

:: --- 迴圈執行：讓 %%a 代表演算法 ID (1 到 9) ---
for /L %%a in (1, 1, 9) do (
    
    :: 自動對應演算法名稱 (僅供顯示輸出)
    set "ALGO_NAME=Unknown"
    if %%a==1 set ALGO_NAME=AE-QTS
    if %%a==2 set ALGO_NAME=DE
    if %%a==3 set ALGO_NAME=PSO
    if %%a==4 set ALGO_NAME=TS
    if %%a==5 set ALGO_NAME=QTS
    if %%a==6 set ALGO_NAME=GA
    if %%a==7 set ALGO_NAME=ABC
    if %%a==8 set ALGO_NAME=WOA
    if %%a==9 set ALGO_NAME=QEA

    set "FILE_PATH=%QUESTION_FOLDER%\tasks_part_%QUESTION_ID%.txt"
    
    if exist "!FILE_PATH!" (
        echo [啟動] !ALGO_NAME! 正在處理題目 %QUESTION_ID%...
        
        :: 執行指令：傳入演算法 %%a 與 檔案路徑
        start /min "!ALGO_NAME!_Task_%QUESTION_ID%" powershell -Command python -u %PYTHON_FILE% %%a "!FILE_PATH!"
    ) else (
        echo [跳過] 找不到檔案: !FILE_PATH!
    )
    
    :: 稍微延遲避免瞬間啟動過多進程導致系統卡頓
    timeout /t 1 >nul
)

echo.
echo [完成] 演算法 1~9 已全部啟動。
pause