import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from sklearn.metrics import roc_auc_score

def calculate_auc(y_true, y_pred):
    # Check if there are at least two unique values in y_true
    if len(np.unique(y_true)) < 2:
        return None  # or you can return 0 or another default value
    return roc_auc_score(y_true, y_pred)

def process_file(file_path, algorithm_name, plotcolor, operationflag):
    print(operationflag)
    print(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    auc_scores = []  # To store AUC scores at each step

    if operationflag == 1:
        
        # Read data
        data = pd.read_csv(file_path, skiprows=1, header=None, delimiter=",")
        y_true, y_pred = data.iloc[:, 0], data.iloc[:, 1]
        
        # Calculate AUC incrementally for each line
        for i in range(1, len(data) + 1):
            current_y_true = y_true[:i]
            current_y_pred = y_pred[:i]
            auc = calculate_auc(current_y_true, current_y_pred)
                        
            # Append the AUC score or a placeholder value (e.g., None or 0) when AUC is undefined
            auc_scores.append(auc if auc is not None else 0)
        
        # # Filter out any None or empty values from auc_scores
        # auc_scores = [score for score in auc_scores if score is not None]

        # Save AUC scores to a CSV file
        auc_scores_df = pd.DataFrame(auc_scores, columns=["AUC"])
        auc_scores_filename = f"auc_scores_{algorithm_name}.txt"
        auc_scores_df.to_csv(auc_scores_filename, index=False)
        print(f"AUC scores saved as {auc_scores_filename}")


    elif operationflag == 2:
        auc_scores = pd.read_csv(file_path, skiprows=1)
        print("AUC Scores loaded from the file:")

    # Plot the AUC scores
    plt.figure(figsize=(10, 6))
    plt.plot(auc_scores, label='AUC', color=plotcolor, marker='o')
    plt.ylim(0.4, 1.0)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))


    # Add labels and title
    # plt.xlabel('Data Points', fontsize=40)
    # plt.ylabel('Cumul. AUC', fontsize=40)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=40)  # Set font size for both x and y axis tick labels

    # Save the plot as a PNG file in the current folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'auc_{algorithm_name}_{current_time}.png'

    # Reduce white space around the figure
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"AUC plot saved as {plot_filename}")
        
    # # Print final results
    # final_auc = auc_scores.iloc[-1]
    # print(f"Final AUC: {final_auc:.3f} ({file_path})")

def main(algorithm_name, file_path, plotcolor, operationflag):
    process_file(file_path, algorithm_name, plotcolor, operationflag)
    

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python3 script.py <algorithm_name> <file_path>")
    #     sys.exit(1)
    
    algorithm_name = sys.argv[1]
    file_path = sys.argv[2]
    plotcolor = sys.argv[3]
    operationflag = int(sys.argv[4])
    
    print(f"Running script with algorithm_name={algorithm_name}, file_path={file_path}, plotcolor={plotcolor}, operationflag={operationflag}")
    
    main(algorithm_name, file_path, plotcolor, operationflag)
