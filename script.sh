#!/bin/bash

CONFIG_FILE="configs/directories.txt"
PYTHON_SCRIPT="pipeline.py"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Read each line from the config file and execute the Python script
while IFS= read -r dir; do
    if [[ -n "$dir" ]]; then
        echo "Processing directory: $dir"
        python "$PYTHON_SCRIPT" -d "$dir"
    fi
done < "$CONFIG_FILE"
