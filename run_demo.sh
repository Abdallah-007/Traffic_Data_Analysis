#!/bin/bash

# Script to run the Chicago Traffic Crashes Analysis Streamlit demo

echo "Starting Chicago Traffic Crashes Analysis Demo..."
echo "---------------------------------------------------"

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit is not installed. Installing required packages..."
    pip install streamlit pandas numpy matplotlib seaborn plotly pillow
fi

# Set the path to the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Run the Streamlit app
echo "Launching Streamlit app..."
streamlit run traffic_crash_demo.py

# If Streamlit exits, print a message
echo "---------------------------------------------------"
echo "Streamlit app closed. Thank you for using the Chicago Traffic Crashes Analysis Demo!" 