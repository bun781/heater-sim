@echo off
REM Windows batch script to run the project

echo 🧮 ODE Solver ^& Beautiful Graphs Toolkit
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing required packages...
pip install -r requirements.txt

REM Run the main script
echo 🚀 Starting the application...
python run_examples.py

pause
