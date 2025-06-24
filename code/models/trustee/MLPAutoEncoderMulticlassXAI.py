import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

import numpy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from tensorflow.keras.losses import MeanSquaredError

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.trustee.IAnomalyDetectionModelXAI import IAnomalyDetectionModelXAI
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


class MLPAutoEncoderMulticlassModelXAI(IAnomalyDetectionModelXAI):
    """
    Multi-layer perceptron (MLP) based autoencoder.
    """

    def __init__(
        self,
        filter_label: Optional[int] = None,
        **kwargs,
    ):
        """
        TODO: multi-AE setup where PredictionField.GROUND_TRUTH label of the data
         will be used to split the data into multiple training sets, then training
         multiple AEs to each predict their own class only.

        :param filter_label:
        :param kwargs: Arguments for the superclass constructor.
        """
        self.model_instances = {}
        self.scaler = StandardScaler()
        self.thresholds = {}
        self.filter_label = filter_label
        super().__init__(**kwargs)

    def get_reconstruction_error(self, model, x):
        r = model.predict(x)
        return MeanSquaredError().call(x, r)

    def train(
        self,
        data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None],
        **kwargs,
    ):
        log.info("Training multiple MLP autoencoders.")
        data_prep_time = 0

        all_encoded_features = []
        encoded_features_by_ground_truth = {}

        for samples, encodings in data:
            start = time.process_time_ns()
            if isinstance(samples, list):
                for sample, encoding in zip(samples, encodings):
                    all_encoded_features.append(encoding)
                    if sample[PredictionField.GROUND_TRUTH] not in encoded_features_by_ground_truth:
                        encoded_features_by_ground_truth[sample[PredictionField.GROUND_TRUTH]] = [encoding]
                    else:
                        encoded_features_by_ground_truth[sample[PredictionField.GROUND_TRUTH]].append(encoding)
            else:
                all_encoded_features.append(encodings[0])
                if samples[PredictionField.GROUND_TRUTH] not in encoded_features_by_ground_truth:
                    encoded_features_by_ground_truth[samples[PredictionField.GROUND_TRUTH]] = encodings
                else:
                    encoded_features_by_ground_truth[samples[PredictionField.GROUND_TRUTH]].append(encodings[0])
            data_prep_time += time.process_time_ns() - start

        self.scaler.fit(all_encoded_features)
        training_start = time.process_time_ns()
        self.thresholds = {}
        for ground_truth in encoded_features_by_ground_truth:
            # TODO make model parameters configurable.
            self.model_instances[ground_truth] = MLPRegressor(
                alpha=1e-15,
                hidden_layer_sizes=[
                    25,
                    50,
                    25,
                    2,
                    25,
                    50,
                    25,
                ],
                random_state=1,
                max_iter=10000,
            )
            encoded_features = encoded_features_by_ground_truth[ground_truth]
            scaled_encoded_features = self.scaler.transform(encoded_features)
            self.model_instances[ground_truth].fit(scaled_encoded_features, scaled_encoded_features)
            train_loss = MeanSquaredError().call(self.model_instances[ground_truth].predict(scaled_encoded_features), np.array(scaled_encoded_features))
            self.thresholds[ground_truth] = np.mean(train_loss.numpy()) + np.std(train_loss.numpy())

        training_time = time.process_time_ns() - training_start
        sample_count = sum([len(encoded_features_by_ground_truth[ground_truth]) for ground_truth in encoded_features_by_ground_truth])

        report_performance(type(self).__name__ + "-preparation", log, sample_count,
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, sample_count,
                           training_time)

        if not self.skip_saving_model:
            dump({"models": self.model_instances, "thresholds": self.thresholds}, self.store_file)
            dump(self.scaler, f'{self.store_file}-scaler')


    def load(self):
        print(self.store_file)
        model_info = load(self.store_file)
        if not model_info:
            log.error(f"Failed to load model from: {self.store_file}")
            return

        self.model_instances = model_info['models']
        self.thresholds = model_info['thresholds']
        if not self.model_instances or self.thresholds:
            log.error(f"Failed to load model from: {self.store_file}")

        self.scaler = load(f'{self.store_file}-scaler')
        if not self.scaler:
            log.error(f"Fail to load scaler for model from {self.store_file}-scaler")

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        sum_processing_time = 0
        sum_samples = 0
        trained_classes = [ground_truth for ground_truth in self.model_instances]

        for sample, encoded_sample in data:
            start_time_ref = time.process_time_ns()
            predictions = {}
            scaled_encoded_sample = self.scaler.transform(encoded_sample)
            for ground_truth in self.model_instances:
                predictions[ground_truth] = self.model_instances[ground_truth].predict(scaled_encoded_sample)

            if isinstance(sample, list):
                # Handle the prediction for multi-sample encoding.
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    sample_predictions = [predictions[ground_truth][i] for ground_truth in predictions]
                    sample_mse = [MeanSquaredError().call(scaled_encoded_sample[i], sample_prediction).numpy() for sample_prediction in sample_predictions]
                    s[PredictionField.OUTPUT_CLASS] = trained_classes[np.argmin(sample_mse)]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s

            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample_predictions = [predictions[ground_truth][0] for ground_truth in predictions]
                sample_mse = [MeanSquaredError().call(scaled_encoded_sample, sample_prediction).numpy() for
                              sample_prediction in sample_predictions]
                sample[PredictionField.OUTPUT_CLASS] = trained_classes[np.argmin(sample_mse)]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)

    def get_training_data(self):
        return self.encoded_features

    def get_labels_name(self):
        return self.feature_names

    def get_test_data(self):
        return self.data_set

    def predict_proba(self, data):
        pass
        # return self.model_instance.predict_proba(data)