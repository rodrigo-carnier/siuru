import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, auc

def process_folder(folder_path, algorithm_name):
    metrics_list = []
    filenames = []  # List to store filenames corresponding to the runs
    
    print(f"Processing folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Processing TXT files
            file_path = os.path.join(folder_path, filename)
            print(f"Reading file: {file_path}")
            data = pd.read_csv(file_path, skiprows=1, header=None, delimiter=",")  # Explicit comma delimiter
            y_true, y_pred = data.iloc[:, 0], data.iloc[:, 1]
            metrics = calculate_metrics(y_true, y_pred)
            metrics_list.append(metrics)
            filenames.append(filename)  # Store the filename corresponding to the run
    
    if not metrics_list:
        print("No valid files found in the folder.")
        return
    
    metrics_array = np.array(metrics_list)
    means = np.mean(metrics_array, axis=0)
    stds = np.std(metrics_array, axis=0)
    
    best_run_index = np.argmax(metrics_array[:, 4])
    worst_run_index = np.argmin(metrics_array[:, 4])
    
    best_run = metrics_array[best_run_index]
    worst_run = metrics_array[worst_run_index]
    
    best_run_filename = filenames[best_run_index]  # Get the filename for the best run
    worst_run_filename = filenames[worst_run_index]  # Get the filename for the worst run
    
    # Generating LaTeX table
    
    best_run_short = best_run_filename[:9]
    worst_run_short = worst_run_filename[:9]
    
    latex_table = f"""
    \\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{l c c c c c c c}}
    \\toprule
    \\multirow{{2}}{{*}}{{\\textbf{{Algorithm}}}} & \\textbf{{Dataset}} & \\textbf{{AUC}} & \\textbf{{Precision}} & \\textbf{{Accuracy}} & \\textbf{{Recall}} & \\textbf{{F1 Score}} & \\textbf{{Bandwidth}} \\\\
    & & \\multicolumn{{5}}{{c}}{{(Mean $\\pm$ Standard Deviation)}} \\\\
    \\midrule
    \\textbf{{{algorithm_name}}} & All & {means[4]:.3f}\% $\\pm$ {stds[4]:.3f}\% & {means[1]*100:.1f}\% $\\pm$ {stds[1]*100:.1f}\% & {means[0]*100:.1f}\% $\\pm$ {stds[0]*100:.1f}\% & {means[2]*100:.1f}\% $\\pm$ {stds[2]*100:.1f}\% & {means[3]*100:.1f}\% $\\pm$ {stds[3]*100:.1f}\% & \\\\
    {algorithm_name} best run & {best_run_short} & {best_run[4]:.3f}\% & {best_run[1]*100:.1f}\% & {best_run[0]*100:.1f}\% & {best_run[2]*100:.1f}\% & {best_run[3]*100:.1f}\% & \\\\
    {algorithm_name} worst run & {worst_run_short} & {worst_run[4]:.3f}\% & {worst_run[1]*100:.1f}\% & {worst_run[0]*100:.1f}\% & {worst_run[2]*100:.1f}\% & {worst_run[3]*100:.1f}\% & \\\\
    \\bottomrule
    \\end{{tabular}}
    \\caption{{Performance metrics for {algorithm_name}.}}
    \\end{{table}}

    \\textbf{{Best Run File:}} {best_run_filename} \\\\
    \\textbf{{Worst Run File:}} {worst_run_filename}
    """
    
    output_file = f"{algorithm_name}_metrics.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")
    print(f"Best run file: {best_run_filename}")
    print(f"Worst run file: {worst_run_filename}")

def main(flag, algorithm_name, folder_path):
    if flag == 1:
        process_folder(folder_path, algorithm_name)

# Command-line argument handling
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 script.py <flag> <algorithm_name> <folder_path>")
        sys.exit(1)

    flag = int(sys.argv[1])
    algorithm_name = sys.argv[2]
    folder_path = sys.argv[3]

    print(f"Running script with flag={flag}, algorithm_name={algorithm_name}, folder_path={folder_path}")

    main(flag, algorithm_name, folder_path)
