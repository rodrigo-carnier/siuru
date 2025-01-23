#!/bin/bash

# Check if a folder path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

# Assign the folder path
FOLDER_PATH="$1"

# Verify if the provided path is a directory
if [ ! -d "$FOLDER_PATH" ]; then
  echo "Error: $FOLDER_PATH is not a directory."
  exit 1
fi

# Iterate over each file in the folder
for file in "$FOLDER_PATH"/*; do
  if [ -f "$file" ]; then
    echo "Processing file: $file"
    python3 code/IoT-AD-stream.py -c "$file"
  fi
done

