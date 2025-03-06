import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

# import numpy
import numpy as np
np.float = float
np.int = np.int32
np.bool = np.bool_

from river import anomaly
from river import compose
from river import metrics
from river import preprocessing
from river import time_series
from river import linear_model
from river import optim


from joblib import dump, load

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


import matplotlib.pyplot as plt
import pickle

from enum import Enum


class PredictiveADModel(IAnomalyDetectionModel):
    """
    Generic interface for anomaly detection model classes to implement.
    """

    def __init__(
        self,
        model_name,
        new_model=True,
        skip_saving_model=False,
        model_storage_base_path=None,
        model_relative_path=None,
        save_interval=3000,  # Interval to save model periodically
        **kwargs,
    ):
        
        period = 12
        predictive_model = time_series.SNARIMAX(
            p=period,
            d=1,
            q=period,
            m=period,
            sd=1,
            regressor=(
                preprocessing.StandardScaler()
                # preprocessing.MinMaxScaler()
                # preprocessing.AdaptativeStandardScaler(fading_factor=.3)
                # preprocessing.RobustScaler()
                | linear_model.LinearRegression(
                    optimizer=optim.SGD(0.005),
                )
            ),
        )


        self.model_instance = anomaly.PredictiveAnomalyDetection(predictive_model, horizon=1, n_std=3.0, warmup_period=25)
        

        
        self.anomaly_threshold = None
        self.trainingScores = None
        self.score_window_size = 25  # Number of scores to store
        self.threshold_coef = 1.0
        self.last_scores = []
        self.save_interval = save_interval
        self.sample_count = 0
        
        super().__init__(
            model_name,
            new_model=new_model,
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

        # Necessary to scale samples, but River only works with dictionaries, so transforming
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
            'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
            'feature11', 'feature12']
        encoded_features = [dict(zip(feature_names, arr)) for arr in encoded_features]
        
        for x, lab in zip(encoded_features, labels):
            # self.scaler.learn_one(x)
            # x = self.scaler.transform_one(x)  # Scale the features
            self.model_instance.learn_one(x, lab) # After scaling, learn

        training_time = time.process_time_ns() - training_start

        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)

        if not self.skip_saving_model:
            self._save_model()

    def _save_model(self):
        dump(self.model_instance, self.store_file)
        log.info(f"Model saved to {self.store_file}")

    def load(self, **kwargs):
        self.model_instance = load(self.store_file)
        if not self.model_instance:
            log.error(f"Failed to load model from: {self.store_file}")

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        sum_processing_time = 0
        sum_samples = 0
        
        self.anomaly_threshold = 0.2
        # self.anomaly_threshold = 0.873

        i=0
        for sample, encoded_sample in data:
            i = i+1
            start_time_ref = time.process_time_ns()

            # Necessary to scale samples, but River only works with dictionaries, so transforming
            feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                'feature11', 'feature12']
            encoded_sample = [dict(zip(feature_names, arr)) for arr in encoded_sample]
            # print(f"The encoded features are {encoded_sample}")

            for x in encoded_sample:
                score = self.model_instance.score_one(x, None)  # Use None for the feature input if not applicable
                self.model_instance.learn_one(x, None)  # Still update the model without the actual label

                # Update last_scores and maintain only the last `score_window_size` scores
                self.last_scores.append(score)
                if len(self.last_scores) > self.score_window_size:
                    self.last_scores.pop(0)  # Remove the oldest score to keep the size consistent

                # Calculate the anomaly threshold as 20% of the mean of the last scores
                if self.last_scores:
                    self.anomaly_threshold = self.threshold_coef * (sum(self.last_scores) / len(self.last_scores))
                    # self.anomaly_threshold = 0.2
                

                if score > self.anomaly_threshold: # Have to decide label using threshold
                    prediction = 1
                else:
                    prediction = 0
                self.model_instance.learn_one(x)

                self.sample_count += 1
                if self.sample_count % self.save_interval == 0 and not self.skip_saving_model:
                    self._save_model()

            if i<25:
                print(score, prediction)
            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i]
                    s[PredictionField.ANOMALY_SCORE] = score[i]
                    s[PredictionField.ANOMALY_THRESHOLD] = self.anomaly_threshold[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction
                sample[PredictionField.ANOMALY_SCORE] = score
                sample[PredictionField.ANOMALY_THRESHOLD] = self.anomaly_threshold
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample
        
        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)
    