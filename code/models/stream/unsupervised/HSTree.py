import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

# import numpy
import numpy as np
np.float = float
np.int = np.int32
np.bool = np.bool_

from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing

from joblib import dump, load

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


import matplotlib.pyplot as plt
import pickle

from enum import Enum


class HSTreeModel(IAnomalyDetectionModel):
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
        
        self.model_instance = anomaly.HalfSpaceTrees(n_trees=2, height=6, window_size=2000, seed=42)

        # self.model_instance = anomaly.HalfSpaceTrees(n_trees=10, height=5, window_size=2000, seed=42)
        # [[1004 3517]         [   0  955]]



        # self.model_instance = anomaly.HalfSpaceTrees(n_trees=3, height=5, window_size=2000, seed=42)
        # [[1619 2902]  [   0  955]]
        self.anomaly_threshold = None
        # self.model_instance = anomaly.HalfSpaceTrees(seed=42)
        # self.model_instance = compose.Pipeline(preprocessing.MinMaxScaler(),anomaly.HalfSpaceTrees(n_trees=5, height=3, window_size=3, seed=42))
        # self.auc = metrics.ROCAUC()
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
        
        log.info("Pretraining method for stream-data unsupervised model HalfSpace Tree.")

        data_prep_time = 0

        single_array_processing = False
        concatenated_data_array = None

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

        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
            'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
            'feature11', 'feature12']

        encoded_features = [dict(zip(feature_names, arr)) for arr in encoded_features]
        
        scaler = preprocessing.MinMaxScaler()
        for x in encoded_features:
            scaler.learn_one(x)
            x = scaler.transform_one(x)  # Scale the features
            self.model_instance.learn_one(x)

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

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        sum_processing_time = 0
        sum_samples = 0
        
        self.anomaly_threshold = 0.873
        for sample, encoded_sample in data:
            start_time_ref = time.process_time_ns()

            feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                'feature11', 'feature12']

            encoded_sample = [dict(zip(feature_names, arr)) for arr in encoded_sample]
            scaler = preprocessing.MinMaxScaler()
            for x in encoded_sample:
                score = self.model_instance.score_one(x)
                if score > self.anomaly_threshold:
                    prediction = 1
                else:
                    prediction = 0
                scaler.learn_one(x)
                x = scaler.transform_one(x)  # Scale the features
                self.model_instance.learn_one(x)
            print(score, prediction)
            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i]
                    s[PredictionField.ANOMALY_SCORE] = score[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction
                sample[PredictionField.ANOMALY_SCORE] = score
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample
        
        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)
    