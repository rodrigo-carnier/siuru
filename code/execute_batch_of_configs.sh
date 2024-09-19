#!/bin/bash

# Check if a folder is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

# Assign the folder parameter to a variable
folder="$1"

# Loop through all files in the specified folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        python3 code/IoT-AD-stream.py -c "$file"
    fi
done
