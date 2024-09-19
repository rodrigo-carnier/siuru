from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle


class AccuracyReporter(IReporter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ground_truths = []
        self.predicted_labels = []

    def report(self, features: Dict[IFeature, Any]):
        self.ground_truths.append(features[PredictionField.GROUND_TRUTH])
        self.predicted_labels.append(features[PredictionField.OUTPUT_BINARY])
        # self.predicted_labels.append(features[PredictionField.OUTPUT_CLASS])

    def end_processing(self):
        log = PipelineLogger.get_logger()
        labels = sorted(set(self.ground_truths + self.predicted_labels))
        print(labels)


        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(self.ground_truths, self.predicted_labels, labels=labels)
        # # Swap (1, 1) with (2, 2)
        # cnf_matrix[0, 0], cnf_matrix[0, 1] = cnf_matrix[0, 1], cnf_matrix[0, 0]
        # # Swap (1, 2) with (2, 1)
        # cnf_matrix[1, 1], cnf_matrix[1, 0] = cnf_matrix[1, 0], cnf_matrix[1, 1]



        caseclass = 1;
        caseanom = 3;
        labelsName = ["Benign", "Malicious"]
        # labelsName = ["Benign", "Bruteforce"]
        # labelsName = ["Benign", "MalariaDOS"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS", "Flood", "SlowITE", "Malformed"]
        
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


###########################################################################
        
        # OUTPUT FILES

        # Generate the file name with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = 'configurations/zplots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imagepkl_path = os.path.join(output_dir, f'{current_time}_confusion_matrix.pkl')
        image_path = os.path.join(output_dir, f'{current_time}_confusion_matrix.png')
        imagetime_path = os.path.join(output_dir, f'{current_time}_timeseriesanomaly.png')
        image_roc_auc_path = os.path.join(output_dir, f'{current_time}_roc_auc.png')
        text_path = os.path.join(output_dir, f'{current_time}_features_labels.txt')
        
        # Create text file with results of confusion matrix
        with open(imagepkl_path, 'wb') as file:
            pickle.dump(cnf_matrix, file)

        with open(text_path, "w") as file:
            
            # Print headers for readability (optional)
            file.write("Label\n")
            file.write("-" * 20 + "\n")
            
            # Iterate over the range of the maximum length
            for i in range(len(self.predicted_labels)):
                # Get elements or default to empty if the vector is shorter
                elem1 = self.predicted_labels[i]
                
                # Write elements side by side with a tab separator
                file.write(f"{elem1}\n")


        ### PERFORMANCE METRICS
        TN, FP, FN, TP = cnf_matrix.ravel()
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = (TP)/(TP+FP)
        recall = (TP)/(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)
        # accuracy = accuracy_score(self.ground_truths, self.predicted_labels)
        # precision = precision_score(self.ground_truths, self.predicted_labels, average='macro')
        # recall = recall_score(self.ground_truths, self.predicted_labels, average='macro')
        # f1 = f1_score(self.ground_truths, self.predicted_labels, average='macro')

        
        ###########################################################################
        # FIGURE OF CONFUSION MATRIX WITH PERFORMANCE METRICS
        # plt.figure(figsize=(8, 8))  # Increase the height to make room for the text
        # # plt.figure(figsize=(8, 5))
        # sns.set(font_scale=2.5)
        # sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labelsName, yticklabels=labelsName)
        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.93)  # Set to 1 to remove the top margin
        # plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin to make room for x-axis label
        
        #         # plt.title(f'Confusion Matrix:',switch_caseclass(caseclass), switch_caseanom(caseanom))
        # # title = f'{switch_caseclass(caseclass)}. {switch_caseanom(caseanom)}.'
        # # plt.title(title)
        # # Save the plot to a file
        
        # metrics_text = (
        #     f"Accuracy: {accuracy:.2f}\n"
        #     f"Precision: {precision:.2f}\n"
        #     f"Recall: {recall:.2f}\n"
        #     f"F1 Score: {f1:.2f}"
        # )
        # plt.figtext(0.5, 0.01, metrics_text, ha='center', va='top', fontsize=18, wrap=True)
        # # plt.text(0.5, -0.2, metrics_text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=18)
        
        # plt.savefig(image_path)
        # plt.close()  # Close the figure to free up memory


        ###########################################################################

        ##### CREATE A FIGURE WITH GRIDSPEC
        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])  # 2 rows, 1 column

        ##### CONFUSION MATRIX PLOT
        ax1 = fig.add_subplot(gs[0])
        sns.set(font_scale=2.5)
        sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labelsName, yticklabels=labelsName, ax=ax1)
        ax1.set_xlabel('Predicted Labels', fontsize=20)
        ax1.set_ylabel('True Labels', fontsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20)

        ##### PERFORMANCE METRICS TEXT
        ax2 = fig.add_subplot(gs[1])
        metrics_text = (
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall:      {recall * 100:.2f}%\n"
        f"F1 Score:  {f1 * 100:.2f}%"
        )
        ax2.text(0, 0, metrics_text, ha='left', va='center', fontsize=24)
        ax2.axis('off')  # Hide the axis

        # Save the plot to a file
        plt.savefig(image_path)
        plt.close()  # Close the figure to free up memory


        ###########################################################################
        
        # PRINTING A TIME-SERIES ANOMALY DETECTION

        # Example data
        timestamps = np.arange(0, len(self.predicted_labels))  # Example: time points from 0 to 99
        pred_labels = np.array(self.predicted_labels)

        # Plotting the time-series scores
        plt.figure(figsize=(10, 6))  # Create a figure with a specific size

        plt.plot(timestamps, pred_labels, label='Predictions', color='b', linestyle='-', marker='.', markersize=0.5)

        # Adding labels and title
        plt.xlabel('Sample')
        plt.ylabel('Prediction')
        plt.title('Time-Series Predictions')

        # Adding a grid for better readability
        plt.grid(True)

        # Optional: Highlighting thresholds or specific anomalies
        # Example: Highlight scores above a threshold (e.g., 0.8)
        # threshold = 0.230
        # high_anomalies = self.predicted_labels > threshold
        # plt.plot(timestamps[high_anomalies], self.predicted_labels[high_anomalies], 'ro', label='High Anomalies')
        # plt.plot(y=anomaly_threshold, color='r', linestyle='--', linewidth=1, label='Threshold')

        # Add a legend
        plt.legend()

        # Save the plot to a file (optional)
        plt.savefig(imagetime_path)


        ##### PRINTING CONFUSION MATRIX RESULTS IN STDOUT

        log.info(f"\n---\nReport\n"
                 f"\nConfusion matrix:\n\n{cnf_matrix}\n\n"
                 f"Labels: {labels}\n"
                 f"(i-th row, j-th column: samples with true label i and predicted label j)\n\n"
                 f"Accuracy:"
                 f"{accuracy_score(self.ground_truths, self.predicted_labels)}\n"
                 f"Precision:"
                 f"{precision_score(self.ground_truths, self.predicted_labels)}\n"
                 f"Recall:"
                 f"{recall_score(self.ground_truths, self.predicted_labels)}\n"
                 f"F1 score: "
                 f"{f1_score(self.ground_truths, self.predicted_labels)}\n---"
                 )

        
        # # True labels
        # self.ground_truths = np.array([0] * 2699 + [1] * 2700)  # Combining TN, FP for 0s and FN, TP for 1s

        # # Predicted labels
        # self.predicted_labels = np.array([0] * 2465 + [1] * 234 + [0] * 859 + [1] * 1841)  # Corresponding predicted labels

        # ### PERFORMANCE METRICS
        # accuracy = accuracy_score(self.ground_truths, self.predicted_labels)
        # precision = precision_score(self.ground_truths, self.predicted_labels, average='macro')
        # recall = recall_score(self.ground_truths, self.predicted_labels, average='macro')
        # f1 = f1_score(self.ground_truths, self.predicted_labels, average='macro')

        # log.info(f"Accuracy:"
        #          f"{accuracy_score(self.ground_truths, self.predicted_labels)}\n"
        #          f"Precision:"
        #          f"{precision_score(self.ground_truths, self.predicted_labels, average='macro')}\n"
        #          f"Recall:"
        #          f"{recall_score(self.ground_truths, self.predicted_labels, average='macro')}\n"
        #          f"F1 score: "
        #          f"{f1_score(self.ground_truths, self.predicted_labels, average='macro')}\n---"
        #          )


        ############ ROC AND AUC

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(self.ground_truths, self.predicted_labels)

        # Calculate the AUC
        roc_auc = auc(fpr, tpr)
        # Alternatively, you can use roc_auc_score directly on the true labels and predicted scores
        roc_auc_alternative = roc_auc_score(self.ground_truths, self.predicted_labels)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(image_roc_auc_path)

    @staticmethod
    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_CLASS,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
            
        ]
