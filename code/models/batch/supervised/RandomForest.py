import logging
import time
from typing import Generator, Any, Dict, Tuple

import numpy
import numpy as np
from joblib import dump, load

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


        # ### PRINTING FIGURE IN FILE
        # # Extract the column you want to plot
        # column_to_plot = [[] for _ in range(12)]  # Initialize empty lists for each column

        # for i in range(12):  # Iterate over indices from 0 to 11
        #     column_to_plot[i] = [arr[i] for arr in encoded_features]

        # # Plot the columns
        # plt.plot(column_to_plot[0])
        # plt.title('ip_header_size')
        # plt.savefig('/models/rf-ip_header_size.png')

        # plt.plot(column_to_plot[1])
        # plt.title('ip_data_size')
        # plt.savefig('/models/rf-ip_data_size.png')

        # plt.plot(column_to_plot[2])
        # plt.title('tcp_header_size')
        # plt.savefig('/models/rf-tcp_header_size.png')

        # plt.plot(column_to_plot[3])
        # plt.title('tcp_size')
        # plt.savefig('/models/rf-tcp_size.png')

        # plt.plot(column_to_plot[4])
        # plt.title('tcp_cwr')
        # plt.savefig('/models/rf-tcp_cwr.png')

        # plt.plot(column_to_plot[5])
        # plt.title('tcp_ece')
        # plt.savefig('/models/rf-tcp_ece.png')

        # plt.plot(column_to_plot[6])
        # plt.title('tcp_urg')
        # plt.savefig('/models/rf-tcp_urg.png')

        # plt.plot(column_to_plot[7])
        # plt.title('tcp_ack')
        # plt.savefig('/models/rf-tcp_ack.png')

        # plt.plot(column_to_plot[8])
        # plt.title('tcp_psh')
        # plt.savefig('/models/rf-tcp_psh.png')

        # plt.plot(column_to_plot[9])
        # plt.title('tcp_rst')
        # plt.savefig('/models/rf-tcp_rst.png')

        # plt.plot(column_to_plot[10])
        # plt.title('tcp_syn')
        # plt.savefig('/models/rf-tcp_syn.png')

        # plt.plot(column_to_plot[11])
        # plt.title('tcp_fin')
        # plt.savefig('/models/rf-tcp_fin.png')


        training_start = time.process_time_ns()
        self.model_instance = RandomForestClassifier()
        print(encoded_features)
        print(type(encoded_features))
        self.model_instance.fit(encoded_features, labels)
        training_time = time.process_time_ns() - training_start

        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

    def load(self):
        self.model_instance = load(self.store_file)

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
        for sample, encoded_sample in data:
            start_time_ref = time.process_time_ns()
            prediction = self.model_instance.predict(encoded_sample)
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
