import logging
import time
from typing import Generator, Any, Dict, Tuple

import numpy
import numpy as np
from joblib import dump, load

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel

import matplotlib.pyplot as plt
import pickle


log = logging.getLogger()


class RandomForestModel(IAnomalyDetectionModel):
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
        log.info("Training a random forest classifier.")

        labels = []
        encoded_features = []

        data_prep_time = 0
        for samples, encoding in data:
            start = time.process_time_ns()
            if isinstance(samples, list):
                # Handle the list with multiple samples used together with
                # xarray DataArray encodings.
                for f in samples:
                    labels.append(f[PredictionField.GROUND_TRUTH])
                if len(encoded_features) == 0:
                    encoded_features = encoding
                else:
                    encoded_features = numpy.concatenate(
                        (encoded_features, encoding), axis=0
                    )
            else:
                labels.append(samples[PredictionField.GROUND_TRUTH])
                encoded_features.append(encoding[0])
            data_prep_time += time.process_time_ns() - start

        training_start = time.process_time_ns()
        self.model_instance = RandomForestClassifier()
        # print(encoded_features)
        # print(type(encoded_features))
        self.model_instance.fit(encoded_features, labels)
        training_time = time.process_time_ns() - training_start

        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

    def load(self, **kwargs):
        self.model_instance = load(self.store_file)
        if not self.model_instance:
            log.error(f"Failed to load model from: {self.store_file}")

    def predict(self, data: EncodedSampleGenerator, **kwargs) ->SampleGenerator:
        # Requirements for encoded data:
        #
        # X : {array-like, sparse matrix} of shape (n_samples, n_features)
        #     The input samples. Internally, its dtype will be converted to
        #     ``dtype=np.float32``. If a sparse matrix is provided, it will be
        #     converted into a sparse ``csr_matrix``.
        #
        # Source: https://github.com/scikit-learn/scikit-learn/blob/72a604975102b2d93082385d7a5a7033886cc825/sklearn/ensemble/_forest.py
        sum_processing_time = 0
        sum_samples = 0
        for sample, encoding in data:
            start_time_ref = time.process_time_ns()
            prediction = self.model_instance.predict(encoding)
            if isinstance(sample, list):
                for i, sample in enumerate(sample):
                    sample[PredictionField.MODEL_NAME] = self.model_name
                    sample[PredictionField.OUTPUT_BINARY] = prediction[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield sample
            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction[0]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)


    def evaluate(self, x: list, y: list, **kwargs) -> Tuple[list[int], list[float]]:

        # Perform incremental evaluation on chunks
        progressive_accuracies = []
        cumulative_correct = 0  # To track correct predictions cumulatively
        total_samples = 0  # To track total number of samples seen so far
        cumulative_accuracies = []  # To store cumulative accuracies
        y_pred = []

        chunk_size = 1
        for i in range(0, len(x), chunk_size):
            X_chunk = x[i:i + chunk_size]
            y_chunk = y[i:i + chunk_size]
            
            # Predict on the current chunk
            chunk_predictions = self.model_instance.predict(X_chunk)  # Ensure this returns a list or array
            y_pred.extend(chunk_predictions)  # Extend y_pred with the chunk predictions
            
            # Calculate the accuracy for this chunk
            chunk_correct = sum(p == t for p, t in zip(chunk_predictions, y_chunk))
            
            # Update correct predictions and total samples
            cumulative_correct += chunk_correct
            total_samples += len(y_chunk)
            
            # Calculate cumulative accuracy
            cumulative_accuracy = (cumulative_correct / total_samples) * 100
            cumulative_accuracies.append(cumulative_accuracy)

        # print(cumulative_accuracies)

        # Plot the cumulative accuracy as a time series
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_accuracies, label='Cumulative Accuracy (%)', color='b', linestyle='-', marker='o')
        plt.ylim(60, 100)

        # Add labels and title
        plt.title('Cumulative Accuracy Over Time')
        plt.xlabel('Data Points Processed')
        plt.ylabel('Cumulative Accuracy (%)')
        plt.grid(True)
        plt.legend()

        # Save the plot as a PNG file in the current folder
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'configurations/zplots/{current_time}_{self.model_name}_cumulative_accuracy_plot_batch.png')
        plt.ylim(0, 100)
        plt.savefig(f'configurations/zplots/{current_time}_{self.model_name}_cumulative_accuracy_plot_batch_100.png')

        return y_pred, cumulative_accuracies