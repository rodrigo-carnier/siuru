from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter

from sklearn.cluster import KMeans
import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle


from models.batch.unsupervised.IsolationForest import IsolationForestModel

class OutlierReporter(IReporter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ground_truths = []
        self.inlier_scores = []
        self.anomaly_scores = []
        self.kmeans_labels = []

    def report(self, features: Dict[IFeature, Any]):
        self.ground_truths.append(features[PredictionField.GROUND_TRUTH])
        self.inlier_scores.append(features[PredictionField.OUTPUT_BINARY])
        self.anomaly_scores.append(features[PredictionField.ANOMALY_SCORE])
        # print(features[PredictionField.GROUND_TRUTH])
        # print(features[PredictionField.OUTPUT_BINARY])
        # print(features[PredictionField.ANOMALY_SCORE])
        # print("cheguei aqui 1")

    def end_processing(self):
        log = PipelineLogger.get_logger()
        # self.inlier_scores_nonneg = [0 if x == 1 else 1 if x == -1 else x for x in self.inlier_scores]
        # self.ground_truths_bin = [1 if x > 0 else x for x in self.ground_truths]
        anomaly_scores_array = np.array(self.anomaly_scores).reshape(-1, 1)
        #model.trainingScores = np.array(model.trainingScores).reshape(-1, 1)
        n_clusters = 6  # Set your desired number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(anomaly_scores_array)
        self.predicted_labels = kmeans.labels_

        labels = sorted(set(self.ground_truths + self.predicted_labels))
        print(labels)

        



        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(self.ground_truths, self.predicted_labels, labels=labels)
        # # Swap (1, 1) with (2, 2)
        # cnf_matrix[0, 0], cnf_matrix[0, 1] = cnf_matrix[0, 1], cnf_matrix[0, 0]
        # # Swap (1, 2) with (2, 1)
        # cnf_matrix[1, 1], cnf_matrix[1, 0] = cnf_matrix[1, 0], cnf_matrix[1, 1]



        caseclass = 2;
        caseanom = 4;
        # labelsName = ["Benign", "Malicious"]
        # labelsName = ["Benign", "Bruteforce"]
        # labelsName = ["Benign", "MalariaDOS"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS"]
        # labelsName = ["Benign", "Bruteforce", "MalariaDOS", "Flood", "SlowITE", "Malformed"]
        labelsName = ["Benign", "Bruteforce", "Flood", "MalariaDOS", "Malformed", "SlowITE"]
        
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
        title = f'{switch_caseclass(caseclass)}. {switch_caseanom(caseanom)}.'
        plt.title(title)
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





        # print("cheguei aqui 3")

        # log = PipelineLogger.get_logger()
        
        # labels_bin = set(self.ground_truths_bin + self.inlier_scores_nonneg)
        # labels_bin.add(0)
        # labels_bin = sorted(labels_bin)
        
        # #labels = sorted(set(self.ground_truths.add(0) + self.kmeans_labels))
        # kmeans_labels = self.kmeans_labels.tolist()
        # labels = set(self.ground_truths + kmeans_labels)
        # labels.add(0)
        # labels = sorted(labels)
        
        # cnf_matrix_binary = confusion_matrix(self.ground_truths_bin, self.inlier_scores_nonneg, labels=labels_bin)
        # cnf_matrix = confusion_matrix(self.ground_truths, kmeans_labels, labels=labels)
        # log.info(f"\n---\nReport\n"
        #          f"\nConfusion matrix binary:\n\n{cnf_matrix_binary}\n\n"
        #          f"Labels: {labels_bin}\n"
        #          f"\nConfusion matrix multiclass:\n\n{cnf_matrix}\n\n"
        #          f"Labels: {labels}\n"
        #          f"(i-th row, j-th column: samples with true label i and predicted label j)\n\n"
        #          f"Accuracy:"
        #          f"{accuracy_score(self.ground_truths, self.kmeans_labels)}\n"
        #          f"Precision:"
        #          f"{precision_score(self.ground_truths, self.kmeans_labels, average='macro')}\n"
        #          f"Recall:"
        #          f"{recall_score(self.ground_truths, self.kmeans_labels, average='macro')}\n"
        #          f"F1 score: "
        #          f"{f1_score(self.ground_truths, self.kmeans_labels, average='macro')}\n---"
        #          )

    @staticmethod
    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
            PredictionField.ANOMALY_SCORE,
            
        ]
