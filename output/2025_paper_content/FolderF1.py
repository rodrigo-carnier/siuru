import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import f1_score

def calculate_cumulative_f1(y_true, y_pred_labels):
    if len(np.unique(y_true)) < 2:
        return None
    return f1_score(y_true, y_pred_labels, zero_division=0)

def process_file(file_path, plotcolor, operationflag):
    print(operationflag)
    print(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    f1_scores = []

    dir_name = os.path.dirname(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    if operationflag == 1:
        # Read CSV with true labels and predicted labels
        data = pd.read_csv(file_path, skiprows=1, header=None, delimiter=",")
        y_true_full, y_pred_labels_full = data.iloc[:, 0], data.iloc[:, 1]
        
        for i in range(1, len(data) + 1):
            y_true_partial = y_true_full[:i]
            y_pred_partial = y_pred_labels_full[:i]
            f1 = calculate_cumulative_f1(y_true_partial, y_pred_partial)
            f1_scores.append(f1 if f1 is not None else 0)

        # Save F1 scores as-is (0-1 range)
        f1_scores_df = pd.DataFrame(f1_scores, columns=["F1"])
        
        f1_scores_filename = os.path.join(dir_name, f"{base_filename}_f1score.txt") if dir_name else f"{base_filename}_f1score.txt"
        f1_scores_df.to_csv(f1_scores_filename, index=False)
        print(f"F1 scores saved as {f1_scores_filename}")

    elif operationflag == 2:
        # Load F1 scores from file (assumed to be 0-1)
        f1_scores_df = pd.read_csv(file_path, skiprows=1)
        f1_scores = f1_scores_df.iloc[:, 0].tolist()
        print("F1 Scores loaded from the file:")

    # Plotting the cumulative F1 scores (0-1 range)
    plt.figure(figsize=(10, 6))
    plt.plot(f1_scores, label='Cumulative F1 Score', color=plotcolor, marker='o')
    plt.ylim(0.5, 1.0)

    # Format Y-axis to show decimals from 0.0 to 1.0 stepping by 0.1
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Format X-axis ticks (thousands as K)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

    plt.grid(True)
    plt.tick_params(axis='both', labelsize=40)
    # Optional labels and legend uncomment if needed:
    # plt.xlabel("Samples Processed")
    # plt.ylabel("F1 Score")
    # plt.title(f"Cumulative F1 Score - {algorithm_name}")
    # plt.legend()

    plot_filename = os.path.join(dir_name, f"{base_filename}.png") if dir_name else f"{base_filename}.png"
    
    # Save the figure with minimal white space
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free memory
    print(f"F1 plot saved as {plot_filename}")

def process_folder(folder_path, plotcolor, operationflag):
    if not os.path.isdir(folder_path):
        print(f"Provided path '{folder_path}' is not a folder or does not exist.")
        return

    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        print("No files found in the folder.")
        return

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        process_file(file_path, plotcolor, operationflag)

def main(folder_path, plotcolor, operationflag):
    process_folder(folder_path, plotcolor, operationflag)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 FolderF1.py <folder_path> <plot_color> <operation_flag>")
        sys.exit(1)

    # default blue, red, green, ,orange, purple, brown, black, gray, pink, cyan, olive
    
    folder_path = sys.argv[1]
    plotcolor = sys.argv[2]
    operationflag = int(sys.argv[3])
    
    print(f"Running script with folder_path={folder_path}, plotcolor={plotcolor}, operationflag={operationflag}")
    main(folder_path, plotcolor, operationflag)

