#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_folder> <destination_folder>"
    exit 1
fi

src_dir="$1"
dst_dir="$2"

# Create destination folder if it doesn't exist
mkdir -p "$dst_dir"

index=1

# Find matching files and copy with new names
while IFS= read -r -d '' file; do
    base_name=$(basename "$file")
    prefix=${base_name:0:27}
    new_name="${prefix}_computation_costs_${index}.txt"
    cp "$file" "$dst_dir/$new_name"
    echo "Copied: $file -> $dst_dir/$new_name"
    ((index++))
done < <(find "$src_dir" -type f -name '*_computation_costs.txt' -print0)

