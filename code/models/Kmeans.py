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


import matplotlib.pyplot as plt
import pickle


class KmeansModel(IAnomalyDetectionModel):
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


        # print(encoded_features)


        # ### SAVING TO FILE
        # # Your object
        # your_object = encoded_features

        # # File path to save the object
        # file_path = '/models/encoded_feat.pkl'

        # # Save the object to a file
        # with open(file_path, 'wb') as file:
        #     pickle.dump(your_object, file)



        # ### PRINTING FIGURE IN FILE
        # # Extract the column you want to plot
        # column_to_plot = [[] for _ in range(12)]  # Initialize empty lists for each column

        # for i in range(12):  # Iterate over indices from 0 to 11
        #     column_to_plot[i] = [arr[i] for arr in encoded_features[1:10000]]
        #     print(column_to_plot[i])

        # # Iterate over each element of column_to_plot
        # for i, column in enumerate(column_to_plot):
        #     plt.figure()  # Create a new figure for each plot
        #     plt.plot(column)  # Plot the data
        #     plt.title(f'ip_data_size_{i}')  # Set a unique title for each plot
        #     plt.savefig(f'/models/ip_data_size_{i}.png')  # Save the plot to a file with a unique name
        #     plt.close()  # Close the current figure to free up memory

        # # Plot the columns
        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[0])
        # plt.title('ip_header_size')
        # plt.savefig('/models/ip_header_size.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[1])
        # plt.title('ip_data_size')
        # plt.savefig('/models/ip_data_size.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[2])
        # plt.title('tcp_header_size')
        # plt.savefig('/models/tcp_header_size.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[3])
        # plt.title('tcp_size')
        # plt.savefig('/models/tcp_size.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[4])
        # plt.title('tcp_cwr')
        # plt.savefig('/models/tcp_cwr.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[5])
        # plt.title('tcp_ece')
        # plt.savefig('/models/tcp_ece.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[6])
        # plt.title('tcp_urg')
        # plt.savefig('/models/tcp_urg.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[7])
        # plt.title('tcp_ack')
        # plt.savefig('/models/tcp_ack.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[# plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()

        # # Save the plot to a PNG file
        # plt.savefig('kmeans_plot.png')

        # # Close the plot to avoid displaying it in a window
        # plt.close()8])
        # plt.title('tcp_psh')
        # plt.savefig('/models/tcp_psh.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[9])
        # plt.title('tcp_rst')
        # plt.savefig('/models/tcp# plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()

        # # Save the plot to a PNG file
        # plt.savefig('kmeans_plot.png')

        # # Close the plot to avoid displaying it in a window
        # plt.close()_rst.png')
        # plt.close()  # Close the current figure to free up memory

        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[10])
        # plt.title('tcp_syn')
        # plt.savefig('/models/tcp_syn.png')
        # plt.close()  # Close the current figure to free up memory
# plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()

        # # Save the plot to a PNG file
        # plt.savefig('kmeans_plot.png')

        # # Close the plot to avoid displaying it in a window
        # plt.close()
        # plt.figure()  # Create a new figure for each plot
        # plt.plot(column_to_plot[11])
        # plt.title('tcp_fin')
        # plt.savefig('/models/tcp_fin.png')
        # plt.close()  # Close the current figure to free up memory


# plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()

        # # Save the plot to a PNG file
        # plt.savefig('kmeans_plot.png')

        # # Close the plot to avoid displaying it in a window
        # plt.close()
        ### Specific of clustering 

        # # Define the number of clusters
        # # 1 - benighn
        # # 2 - bruteforce
        # # 3 - flood
        # # 4 - malariados
        # # 5 - malformed
        # # 6 - slowite
        # n_clusters = 6
        
        # Define the number of clusters
        # 1 - benighn
        # 2 - flood
        n_clusters = 6

        self.model_instance = KMeans(n_clusters=n_clusters, random_state=18, n_init=10)

        if not single_array_processing:
            self.model_instance.fit(concatenated_data_array)
        else:
            self.model_instance.fit(encoded_features)
        training_time = time.process_time_ns() - training_start

        # Get cluster assignments for each sample
        cluster_labels = self.model_instance.labels_

        # Get centroids of clusters
        centroids = self.model_instance.cluster_centers_



        #### RMC 2024-04-04: This code is nonsense because how can I know something is an anomaly besides the labels, which I am not using?

        # # Identify samples distant from centroids
        # # Samples in clusters with fewer members or farther from centroids could be considered anomalies
        # anomalies = []

        # # Expected number of samples belonging to each cluster
        # threshold = 1500

        # # Iterate over samples and cluster labels
        # print(cluster_labels)
        # print(len(encoded_features))
        # for feat, label in zip(encoded_features, cluster_labels):
        #     label = int(label)
        #     if len([s for s, l in zip(encoded_features, cluster_labels) if l == label]) < threshold: # Adjust threshold as needed
        #         anomalies.append(feat)
        #     # Alternatively, you can calculate distance from centroids and consider samples with large distances as anomalies
    
        # # Print or process detected anomalies
        # for anomaly in anomalies:
        #     print("Anomaly:", anomaly)

        ### end of specific




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
        with open("/data/kmeansoutput.txt", "w") as file:
            # Print variable names and their values to the file
            print(f"cluster_labels: {cluster_labels}", file=file)
            print(f"centroids: {centroids}", file=file)
            # print(f"anomalies: {anomalies}", file=file)

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
            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_CLASS] = prediction[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_CLASS] = prediction[0]
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