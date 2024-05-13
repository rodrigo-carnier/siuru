from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter

from sklearn.cluster import KMeans
import numpy as np

from models.IsolationForest import IsolationForestModel

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
        
        self.inlier_scores_nonneg = [0 if x == 1 else 1 if x == -1 else x for x in self.inlier_scores]
        self.ground_truths_bin = [1 if x > 0 else x for x in self.ground_truths]
        anomaly_scores_array = np.array(self.anomaly_scores).reshape(-1, 1)
        #model.trainingScores = np.array(model.trainingScores).reshape(-1, 1)
        n_clusters = 2  # Set your desired number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(anomaly_scores_array)
        self.kmeans_labels = kmeans.labels_

        print("cheguei aqui 3")

        log = PipelineLogger.get_logger()
        
        labels_bin = set(self.ground_truths_bin + self.inlier_scores_nonneg)
        labels_bin.add(0)
        labels_bin = sorted(labels_bin)
        
        #labels = sorted(set(self.ground_truths.add(0) + self.kmeans_labels))
        kmeans_labels = self.kmeans_labels.tolist()
        labels = set(self.ground_truths + kmeans_labels)
        labels.add(0)
        labels = sorted(labels)
        
        cnf_matrix_binary = confusion_matrix(self.ground_truths_bin, self.inlier_scores_nonneg, labels=labels_bin)
        cnf_matrix = confusion_matrix(self.ground_truths, kmeans_labels, labels=labels)
        log.info(f"\n---\nReport\n"
                 f"\nConfusion matrix binary:\n\n{cnf_matrix_binary}\n\n"
                 f"Labels: {labels_bin}\n"
                 f"\nConfusion matrix multiclass:\n\n{cnf_matrix}\n\n"
                 f"Labels: {labels}\n"
                 f"(i-th row, j-th column: samples with true label i and predicted label j)\n\n"
                 f"Accuracy:"
                 f"{accuracy_score(self.ground_truths, self.kmeans_labels)}\n"
                 f"Precision:"
                 f"{precision_score(self.ground_truths, self.kmeans_labels, average='macro')}\n"
                 f"Recall:"
                 f"{recall_score(self.ground_truths, self.kmeans_labels, average='macro')}\n"
                 f"F1 score: "
                 f"{f1_score(self.ground_truths, self.kmeans_labels, average='macro')}\n---"
                 )

    @staticmethod
    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
            PredictionField.ANOMALY_SCORE,
            
        ]
