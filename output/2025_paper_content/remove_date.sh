#!/bin/bash

# Ensure the script receives the input folder as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <input_folder>"
  exit 1
fi

input_folder="$1"

# Function to recursively rename files
rename_files() {
  local folder="$1"
  
  # Find all files in the directory and subdirectories
  find "$folder" -type f | while read -r file; do
    # Get the base name of the file
    filename=$(basename "$file")
    
    # Check if the filename starts with the date pattern (YYYYMMDD_HHMMSS_)
    if [[ "$filename" =~ ^[0-9]{8}_[0-9]{6}_ ]]; then
      # Remove the pattern (first 15 characters) from the filename
      new_filename="${filename:16}"
      
      # Get the full path to the file's folder
      file_folder=$(dirname "$file")
      
      # Get the full path to the renamed file (keeping it in the same folder)
      new_filepath="$file_folder/$new_filename"
      
      # Rename the file by moving it to the new name in the same folder
      mv "$file" "$new_filepath"
      echo "Renamed: $file -> $new_filepath"
    fi
  done
}

# Start renaming files from the input folder
rename_files "$input_folder"

