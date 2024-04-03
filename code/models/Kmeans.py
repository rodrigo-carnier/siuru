import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

import numpy
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()



class Kmeans(IAnomalyDetectionModel):
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
        
        log.info("Training a K-means cluster.")

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


        ### Specific of clustering 

        labels = []

        # Define the number of clusters
        # 1 - benighn
        # 2 - bruteforce
        # 3 - flood
        # 4 - malariados
        # 5 - malformed
        # 6 - slowite
        n_clusters = 6

        self.model_instance = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)

        ### end of specific?

        if not single_array_processing:
            self.model_instance.fit(concatenated_data_array, concatenated_data_array)
        else:
            self.model_instance.fit(encoded_features, encoded_features)
        training_time = time.process_time_ns() - training_start


        # Get cluster assignments for each sample
        cluster_labels = self.model_instance.labels_

        # Get centroids of clusters
        centroids = self.model_instance.cluster_centers_

        # Identify anomalies based on cluster membership or distance from centroids
        # For example, samples in clusters with fewer members or farther from centroids could be considered anomalies
        anomalies = []

        # Expected number of samples belonging to each cluster
        threshold = 1500

        # Iterate over samples and cluster labels
        print(cluster_labels)
        for sample, label in zip(samples, cluster_labels):
            label = int(label)
            if len([s for s, l in zip(samples, cluster_labels) if l == label]) < threshold: # Adjust threshold as needed
                anomalies.append(sample)
            # Alternatively, you can calculate distance from centroids and consider samples with large distances as anomalies

    
        # Print or process detected anomalies
        for anomaly in anomalies:
            print("Anomaly:", anomaly)

        # You can also use the model to predict clusters for new data
        # new_data_clusters = kmeans.predict(new_data)


        sample_count = len(encoded_features) if encoded_features else len(concatenated_data_array)

        report_performance(type(self).__name__ + "-preparation", log, sample_count,
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, sample_count,
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

        # Open a file in write mode
        with open("/data/kmeansoutput.txt", "w") as file:
            # Print variable names and their values to the file
            print(f"cluster_labels: {cluster_labels}", file=file)
            print(f"centroids: {centroids}", file=file)
            print(f"anomalies: {anomalies}", file=file)

    def load(self, **kwargs):
        self.model_instance = load(self.store_file)
        if not self.model_instance:
            log.error(f"Failed to load model from: {self.store_file}")

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        """
        Adds a prediction entry based on encoded data directly into
        the feature dictionary of the provided sample, then return the sample.
        """
        pass
