import time
from typing import Generator, Any, Dict, Tuple, Optional

import numpy
import numpy as np
from joblib import dump, load

from sklearn.ensemble import IsolationForest

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


class IsolationForestMulticlassModel(IAnomalyDetectionModel):
    def __init__(
            self,
            model_name,
            train_new_model=True,
            skip_saving_model=False,
            model_storage_base_path=None,
            model_relative_path=None,
            filter_label: Optional[int] = None,
            **kwargs,
    ):
        self.model_instances = {}
        self.filter_label = filter_label
        self.encoded_features = []
        self.train_data = {}
        self.feature_names = []
        self.data_set = []
        self.test_data = {}
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
        log.info("Training an IsolationForestMulticlass model.")
        data_prep_time = 0

        encoded_features_by_ground_truth = {}

        for samples, encodings in data:
            start = time.process_time_ns()
            if isinstance(samples, list):
                for sample, encoding in zip(samples, encodings):
                    if sample[PredictionField.GROUND_TRUTH] not in encoded_features_by_ground_truth:
                        encoded_features_by_ground_truth[sample[PredictionField.GROUND_TRUTH]] = [encoding]
                    else:
                        encoded_features_by_ground_truth[sample[PredictionField.GROUND_TRUTH]].append(encoding)
            else:
                if samples[PredictionField.GROUND_TRUTH] not in encoded_features_by_ground_truth:
                    encoded_features_by_ground_truth[samples[PredictionField.GROUND_TRUTH]] = encodings
                else:
                    encoded_features_by_ground_truth[samples[PredictionField.GROUND_TRUTH]].append(encodings[0])
            data_prep_time += time.process_time_ns() - start

        training_start = time.process_time_ns()
        for ground_truth in encoded_features_by_ground_truth:
            # TODO make model parameters configurable.
            self.model_instances[ground_truth] = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)
            self.model_instances[ground_truth].fit(encoded_features_by_ground_truth[ground_truth])

        training_time = time.process_time_ns() - training_start

        sample_count = sum([len(encoded_features_by_ground_truth[ground_truth]) for ground_truth in encoded_features_by_ground_truth])

        report_performance(type(self).__name__ + "-preparation", log, sample_count,
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, sample_count,
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instances, self.store_file)

    def load(self):
        self.model_instances = load(self.store_file)

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        sum_processing_time = 0
        sum_samples = 0
        trained_classes = [ground_truth for ground_truth in self.model_instances]
        count = 0
        for sample, encoded_sample in data:
            start_time_ref = time.process_time_ns()
            scores = {}
            # print(count, "decision function")
            for ground_truth in self.model_instances:
                scores[ground_truth] = self.model_instances[ground_truth].decision_function(encoded_sample)

            # print(count, "loop")
            if isinstance(sample, list):
                # Handle the prediction for multi-sample encoding.
                # print(count, "list")
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    sample_scores = [scores[ground_truth][i] for ground_truth in scores]
                    s[PredictionField.OUTPUT_CLASS] = trained_classes[np.argmax(sample_scores)]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s

            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample_scores = [scores[ground_truth][0] for ground_truth in scores]
                sample[PredictionField.OUTPUT_CLASS] = trained_classes[np.argmax(sample_scores)]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

            # print(count, "end")
            # return

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