import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

import numpy
import numpy as np
from sklearn.ensemble import IsolationForest
from joblib import dump, load

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


import matplotlib.pyplot as plt
import pickle


class IsolationForestModel(IAnomalyDetectionModel):
    """
    Generic interface for anomaly detection model classes to implement.
    """

    def __init__(
        self,
        model_name,
        train_new_model=True,
        skip_saving_model=False,
        model_storage_base_path=None,
        model_relative_path=None,
        **kwargs,
    ):
        self.model_instance = None
        self.trainingScores = None
        super().__init__(
            model_name,
            train_new_model=train_new_model,
            skip_saving_model=skip_saving_model,
            model_storage_base_path=model_storage_base_path,
            model_relative_path=model_relative_path,
            **kwargs,
        )


    def train(
        self,
        data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None],
        **kwargs,
        ):
        
        log.info("Training an Isolation Forest.")

        data_prep_time = 0

        single_array_processing = False
        concatenated_data_array = None
        encoded_features = []

        for samples, encoding in data:
            start = time.process_time_ns()
            if isinstance(samples, list):
                if self.filter_label:
                    # TODO filter xarray by GROUND_TRUTH filter.
                    pass
                elif concatenated_data_array is None:
                    concatenated_data_array = encoding
                else:
                    concatenated_data_array = numpy.concatenate(
                        (concatenated_data_array, encoding),
                        axis=0,
                    )
            else:
                single_array_processing = True
                encoded_features.append(encoding[0])
            data_prep_time += time.process_time_ns() - start


        training_start = time.process_time_ns()
        self.model_instance = IsolationForest(random_state=42)

        if not single_array_processing:
            self.model_instance.fit(concatenated_data_array)
            # Calculate the average anomaly score of the training sample(s). CAUTION: for neighbors.LocalOutlierFactor, use decision_function only on new data
            self.trainingScores = self.model_instance.decision_function(concatenated_data_array)
        else:
            self.model_instance.fit(encoded_features)
            # Calculate the average anomaly score of the training sample(s). CAUTION: for neighbors.LocalOutlierFactor, use decision_function only on new data
            self.trainingScores = self.model_instance.decision_function(encoded_features)
        training_time = time.process_time_ns() - training_start




        sample_count = len(encoded_features) if encoded_features else len(concatenated_data_array)

        report_performance(type(self).__name__ + "-preparation", log, sample_count,
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, sample_count,
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

        # Print results
        # print(cluster_labels[1000:1200])
        # print(np.sum(cluster_labels == 0))
        # print(np.sum(cluster_labels == 1))
        # print(np.sum(cluster_labels == 2))
        # print(np.sum(cluster_labels == 3))
        # print(np.sum(cluster_labels == 4))
        # print(np.sum(cluster_labels == 5))

        # Open a file in write mode
        with open("/data/isolationforest.txt", "w") as file:
            # Print variable names and their values to the file
            print(f"training scores: {self.trainingScores}", file=file)

    def load(self, **kwargs):
        self.model_instance = load(self.store_file)
        if not self.model_instance:
            log.error(f"Failed to load model from: {self.store_file}")

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        sum_processing_time = 0
        sum_samples = 0
        for sample, encoded_sample in data:
            start_time_ref = time.process_time_ns()
            encoded_sample = np.array(encoded_sample, dtype=float)
            prediction = self.model_instance.predict(encoded_sample)
            scores = self.model_instance.decision_function(encoded_sample)
            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i]
                    s[PredictionField.ANOMALY_SCORE] = scores[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction[0]
                sample[PredictionField.ANOMALY_SCORE] = scores[0]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)
        

        # # Get the centroids of the clusters
        # centroids = self.model_instance.cluster_centers_

        # # Plot the centroids
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')

        
        # # Plot the data points with different colors for each cluster
        # for cluster_label in set(prediction):
        #     cluster_data = data[prediction == cluster_label]
        #     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

        # plt.title('KMeans Clustering')
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')

        # # Save the plot to a PNG file
        # plt.savefig('/data/kmeans_plot_scatter.png')

        # # Close the plot to avoid displaying it in a window
        # plt.close()


        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)