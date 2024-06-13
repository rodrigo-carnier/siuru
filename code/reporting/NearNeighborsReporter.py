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


        # Generate the file name with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = 'configurations/zplots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        text_path = os.path.join(output_dir, f'confusion_matrix_{current_time}.pkl')
        image_path = os.path.join(output_dir, f'confusion_matrix_{current_time}.png')
        
        # Create text file with results of confusion matrix
        with open(text_path, 'wb') as file:
            pickle.dump(cnf_matrix, file)


        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labelsName, yticklabels=labelsName)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        # plt.title(f'Confusion Matrix:',switch_caseclass(caseclass), switch_caseanom(caseanom))
        # title = f'{switch_caseclass(caseclass)}. {switch_caseanom(caseanom)}.'
        # plt.title(title)
        # Save the plot to a file
        plt.savefig(image_path)
        plt.close()  # Close the figure to free up memory

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
