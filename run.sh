#!/bin/bash
# Simple bash script to run the project

echo "🧮 ODE Solver & Beautiful Graphs Toolkit"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import numpy, matplotlib, scipy, pandas, seaborn, sympy, sklearn, networkx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip3 install -r requirements.txt
fi

# Run the main script
echo "🚀 Starting the application..."
python3 run_examples.py
