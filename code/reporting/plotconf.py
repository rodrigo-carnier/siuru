from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score, ConfusionMatrixDisplay


import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

if __name__ == "__main__":
    ################################

    caseclass = 2;
    caseanom = 4;
    labels = [0, 1]
    # labels = [0, 1, 2]
    # labels = [0, 1, 2, 3, 4, 5]
    # labelsName = ["Benign", "Malicious"]
    # labelsName = ["Benign", "Bruteforce"]
    # labelsName = ["Benign", "MalariaDOS"]
    # labelsName = ["Benign", "Bruteforce", "MalariaDOS"]
    labelsName = ["Benign", "Bruteforce", "Flood", "MalariaDOS", "Malformed", "SlowITE"]
    # labelsName = ["Benign", "Bruteforce", "MalariaDOS", "Flood", "SlowITE", "Malformed"]
    ################################



    def caseclasstype1():
        return "Binary class"
    def caseclasstype2():
        return "Multiclass"
    casescl = {
        1: caseclasstype1,
        2: caseclasstype2
    }
    def switch_caseclass(case):
        return casescl.get(case, lambda: "Invalid case")()

    def caseanomtype1():
        return "Anomaly (1): Bruteforce"
    def caseanomtype2():
        return "Anomaly (1): MalariaDOS"
    def caseanomtype3():
        return "Anomalies (2): Brute + MalDOS"
    def caseanomtype4():
        return "Anomalies (5): all MQTTset"
    casesan = {
        1: caseanomtype1,
        2: caseanomtype2,
        3: caseanomtype3,
        4: caseanomtype4
    }

    def switch_caseanom(case):
        return casesan.get(case, lambda: "Invalid case")()


    # Generate the file name with current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = 'configurations/zplots/confusion_matrix_20240522_053144'
    text_path = f'{output_dir}.pkl'
    image_path = f'{output_dir}_3.png'

    # Load the confusion matrix from the file
    with open(text_path, 'rb') as file:
        cnf_matrix = pickle.load(file)

    # cnf_matrix = [[2808, 0, 0, 0, 0, 1713],
    #               [0, 3847, 0, 12, 0, 0],
    #               [0, 0, 11, 0, 0, 11],
    #               [0, 353, 0, 560, 20, 0],
    #               [0, 1910, 0, 89, 1167, 0],
    #               [0, 0, 0, 0, 8, 2747]]
    
    # cnf_matrix = [[4521, 0, 0, 0, 0, 0],
    #               [3859, 0, 0, 0, 0, 0],
    #               [11, 8, 0, 1, 2, 0],
    #               [753, 0, 127, 0, 0, 53],
    #               [3107, 0, 26, 0, 0, 33],
    #               [2755, 0, 0, 0, 0, 0]]
    print("Confusion matrix loaded from file:")
    print(cnf_matrix)
    # cnf_matrix_1 = []
    # cnf_matrix_1[0, 0] = cnf_matrix[0, 0]
    # cnf_matrix_1[0, 1] = cnf_matrix[0, 1]
    # cnf_matrix_1[1, 0] = cnf_matrix[1, 0]
    # cnf_matrix_1[1, 1] = cnf_matrix[1, 1]
    

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.0)
    sns.heatmap(cnf_matrix[:6,:6], annot=True, fmt='d', cmap='Blues', xticklabels=labelsName, yticklabels=labelsName)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    # plt.title(f'Confusion Matrix:',switch_caseclass(caseclass), switch_caseanom(caseanom))
    title = f'{switch_caseclass(caseclass)}. {switch_caseanom(caseanom)}.'
    plt.title(title)
    # Save the plot to a file
    plt.savefig(image_path)
    plt.close()  # Close the figure to free up memory

    # print(f"\n---\nReport\n"
    #             f"\nConfusion matrix:\n\n{cnf_matrix}\n\n"
    #             f"Labels: {labels}\n"
    #             f"(i-th row, j-th column: samples with true label i and predicted label j)\n\n"
    #             f"Accuracy:"
    #             f"{accuracy_score(self.ground_truths, self.predicted_labels)}\n"
    #             f"Precision:"
    #             f"{precision_score(self.ground_truths, self.predicted_labels, average='macro')}\n"
    #             f"Recall:"
    #             f"{recall_score(self.ground_truths, self.predicted_labels, average='macro')}\n"
    #             f"F1 score: "
    #             f"{f1_score(self.ground_truths, self.predicted_labels, average='macro')}\n---"
    #             )
