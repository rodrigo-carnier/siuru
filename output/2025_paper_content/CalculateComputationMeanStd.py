import os
import re
import numpy as np

def extract_bandwidth(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if "megabits/second Ethernet traffic bandwidth from processing start" in line:
                match = re.search(r"([\d.]+) megabits/second", line)
                if match:
                    return float(match.group(1))
    return None

def process_folder(folder_path):
    bandwidth_values = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            value = extract_bandwidth(file_path)
            if value is not None:
                bandwidth_values.append(value)

    if not bandwidth_values:
        print("No valid bandwidth values found.")
        return

    mean_val = np.mean(bandwidth_values)
    std_val = np.std(bandwidth_values)

    folder_name = os.path.basename(folder_path.rstrip("/\\"))
    output_filename = f"output_{folder_name}.txt"

    output_text = f"{mean_val:.3f} ± {std_val:.3f}\n"

    with open(output_filename, "w") as f:
        f.write(output_text)

    print(f"Results saved to {output_filename}")

def main():
    folder_path = input("Enter the folder path: ").strip()

    if not os.path.isdir(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    process_folder(folder_path)

if __name__ == "__main__":
    main()

