@echo off
chcp 65001 >nul
title PhotoFilter Pro

cd /d "%~dp0"

echo ========================================
echo   🎨 PhotoFilter Pro — 图片调色批量处理
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到 Python，请先安装 Python 3.9+
    echo    下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install dependencies
echo 📦 检查依赖...
pip install -r requirements.txt -q

:: Start server
echo.
echo 🚀 启动服务器...
echo 📡 浏览器打开 http://localhost:8899
echo.
echo 按 Ctrl+C 停止服务器
echo.

start "" http://localhost:8899
python main.py

pause
