from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score, ConfusionMatrixDisplay

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle


class NearNeighborsReporter(IReporter):
    """
    Can only be used when PredictionField.GROUND_TRUTH is known!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ground_truths = []
        self.predicted_labels = []
        self.anomaly_scores = []

    def report(self, features: Dict[IFeature, Any]):
        self.ground_truths.append(features[PredictionField.GROUND_TRUTH])
        self.predicted_labels.append(features[PredictionField.OUTPUT_BINARY])
        self.anomaly_scores.append(features[PredictionField.ANOMALY_SCORE])

    def end_processing(self):
        print("cheguei aqui 3")
        log = PipelineLogger.get_logger()
        labels = sorted(set(self.ground_truths + self.predicted_labels))

        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(self.ground_truths, self.predicted_labels, labels=labels)
        # # Swap (1, 1) with (2, 2)
        # cnf_matrix[0, 0], cnf_matrix[0, 1] = cnf_matrix[0, 1], cnf_matrix[0, 0]
        # # Swap (1, 2) with (2, 1)
        # cnf_matrix[1, 1], cnf_matrix[1, 0] = cnf_matrix[1, 0], cnf_matrix[1, 1]


        ##### LABELS FOR FIGURES

        caseclass = 1;
        caseanom = 3;
        # labelsName = ["Benign", "Malicious"]
        # labelsName = ["Benign", "Bruteforce"]
        # labelsName = ["Benign", "MalariaDOS"]
        labelsName = ["Benign", "Flood"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS", "Malformed"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS", "Malformed", "SlowITE"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS", "Malformed", "SlowITE", "Flood"]
        
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
            return "Anomaly (1): Flood"
        def caseanomtype4():
            return "Anomalies (2): Brute + MalDOS"
        def caseanomtype5():
            return "Anomalies (5): all MQTTset"
        casesan = {
            1: caseanomtype1,
            2: caseanomtype2,
            3: caseanomtype3,
            4: caseanomtype4,
            5: caseanomtype5
        }
        
        def switch_caseanom(case):
            return casesan.get(case, lambda: "Invalid case")()



        ##### OUTPUT FILES

        # Generate the file name with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = 'configurations/zplots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imagepkl_path = os.path.join(output_dir, f'{current_time}_confusion_matrix.pkl')
        image_path = os.path.join(output_dir, f'{current_time}_confusion_matrix.png')
        imagetime_path = os.path.join(output_dir, f'{current_time}_timeseriesanomaly.png')
        text_path = os.path.join(output_dir, f'{current_time}_features_scores_labels.txt')
        
        # Create text file with results of confusion matrix
        with open(imagepkl_path, 'wb') as file:
            pickle.dump(cnf_matrix, file)

        with open(text_path, "w") as file:
            
            # Print headers for readability (optional)
            file.write("Label\tScore\n")
            file.write("-" * 20 + "\n")
            
            # Iterate over the range of the maximum length
            for i in range(len(self.predicted_labels)):
                # Get elements or default to empty if the vector is shorter
                elem1 = self.predicted_labels[i]
                elem2 = self.anomaly_scores[i]
                
                # Write elements side by side with a tab separator
                file.write(f"{elem1}\t{elem2}\n")



        ##### FIGURE OF CONFUSION MATRIX

        

        plt.figure(figsize=(8, 5))
        sns.set(font_scale=2.5)
        sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labelsName, yticklabels=labelsName)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.subplots_adjust(top=1)  # Set to 1 to remove the top margin
        plt.subplots_adjust(bottom=0.20)  # Adjust bottom margin to make room for x-axis label
        # plt.title(f'Confusion Matrix:',switch_caseclass(caseclass), switch_caseanom(caseanom))
        # title = f'{switch_caseclass(caseclass)}. {switch_caseanom(caseanom)}.'
        # plt.title(title)
        # Save the plot to a file
        plt.savefig(image_path)
        plt.close()  # Close the figure to free up memory



        ##### PRINTING A TIME-SERIES ANOMALY DETECTION

        # Example data
        timestamps = np.arange(0, len(self.predicted_labels))  # Example: time points from 0 to 99
        anomaly_scores = np.array(self.anomaly_scores)

        # Plotting the time-series scores
        plt.figure(figsize=(10, 6))  # Create a figure with a specific size

        plt.plot(timestamps, anomaly_scores, label='Anomaly Scores', color='b', linestyle='-', marker='.', markersize=0.5)

        # Adding labels and title
        plt.xlabel('Sample')
        plt.ylabel('Anomaly Score')
        plt.title('Time-Series Anomaly Scores')

        # Adding a grid for better readability
        plt.grid(True)

        # Optional: Highlighting thresholds or specific anomalies
        # Example: Highlight scores above a threshold (e.g., 0.8)
        threshold = 0.230
        # high_anomalies = anomaly_scores > threshold
        # plt.plot(timestamps[high_anomalies], anomaly_scores[high_anomalies], 'ro', label='High Anomalies')
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label='Threshold')


        # Add a legend
        plt.legend()

        # Save the plot to a file (optional)
        plt.savefig(imagetime_path)

        # Display the plot
        plt.show()



        ##### PRINTING CONFUSION MATRIX RESULTS IN STDOUT

        log.info(f"\n---\nReport\n"
                 f"\nConfusion matrix:\n\n{cnf_matrix}\n\n"
                 f"Labels: {labels}\n"
                 f"(i-th row, j-th column: samples with true label i and predicted label j)\n\n"
                 f"Accuracy:"
                 f"{accuracy_score(self.ground_truths, self.predicted_labels)}\n"
                 f"Precision:"
                 f"{precision_score(self.ground_truths, self.predicted_labels, average='macro')}\n"
                 f"Recall:"
                 f"{recall_score(self.ground_truths, self.predicted_labels, average='macro')}\n"
                 f"F1 score: "
                 f"{f1_score(self.ground_truths, self.predicted_labels, average='macro')}\n---"
                 )

    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
            PredictionField.ANOMALY_SCORE,
            
        ]
