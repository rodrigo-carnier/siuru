import logging
import time
from typing import Generator, Any, Dict, Tuple

import numpy
import numpy as np
from joblib import dump, load

import os
import json

from sklearn.ensemble import RandomForestClassifier
# from trustee.report.trust import TrustReport

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from utils import recurse_tree, recurse_tree_with_classes, recurse_tree_with_classes_alphabet_order, tree_to_bracket_notation
from utils.my_trustee.report.trust import TrustReport

log = logging.getLogger()


class RandomForestMulticlassModel(IAnomalyDetectionModel):
    def __init__(
            self,
            model_name,
            train_new_model=True,
            skip_saving_model=False,
            model_storage_base_path=None,
            model_relative_path=None,
            full_config_json=None,
            **kwargs
    ):
        self.model_instance = None
        self.encoded_features = []
        self.train_data = {}
        self.feature_names = []
        self.data_set = []
        self.test_data = {}
        self.existing_labels = None

        super().__init__(
            model_name,
            train_new_model=train_new_model,
            skip_saving_model=skip_saving_model,
            model_storage_base_path=model_storage_base_path,
            model_relative_path=model_relative_path,
            full_config_json=full_config_json,
            **kwargs,
        )

    def save_labels(self, **kwargs):
        data_sources = json.loads(self.full_config_json).get("DATA_SOURCES", [])

        # Build labels
        labels = {
            int(preprocessor["kwargs"].get("label_value")): preprocessor["kwargs"].get("label_name")
            for data_source in data_sources
            for preprocessor in data_source.get("preprocessors", [])
            if preprocessor.get("class") == "FileLabelProcessor"
        }

        if labels:
            # Building the output file path
            output_file_path = os.path.join(self.model_storage_base_path, self.model_name, "multiclass-labels.json")
            log.debug(output_file_path)
            # Saving labels in a new JSON file
            with open(output_file_path, "w") as json_file:
                json.dump(labels, json_file, indent=4, sort_keys=True)
        else:
            raise RuntimeError("No labels in FileLabelProcessor class")

    def compare_labels(self, **kwargs):
        output_file_path = os.path.join(self.model_storage_base_path, self.model_name, "multiclass-labels.json")

        # Load existed labels from JSON file
        with open(output_file_path, "r") as json_file:
            self.existing_labels = json.load(json_file)

        data_sources = json.loads(self.full_config_json).get("DATA_SOURCES", [])

        # Build test labels
        test_label_values = [
            int(preprocessor["kwargs"].get("label_value"))
            for data_source in data_sources
            for preprocessor in data_source.get("preprocessors", [])
            if preprocessor.get("class") == "FileLabelProcessor"
        ]

        log.info(f"Test Label Values: {test_label_values}")
        log.info(f"Existing Labels: {self.existing_labels.keys()}")

        # Check if any label_value in test_label_values is not in existing_labels
        for label_value in test_label_values:
            if str(label_value) not in self.existing_labels:
                raise ValueError(f"The label value: '{label_value}' in test_labels is not present in existing_labels")

        log.info("All values in test_labels exists in existing_labels.")

    def train(
            self,
            data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None],
            **kwargs,
    ):
        self.save_labels()

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
        self.model_instance.fit(encoded_features, labels)
        training_time = time.process_time_ns() - training_start

        self.encoded_features = np.array(encoded_features)
        self.train_data['X'] = self.encoded_features
        self.train_data['y'] = np.array(labels)

        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

    def load(self):
        self.model_instance = load(self.store_file)

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        self.compare_labels()
        sum_processing_time = 0
        sum_samples = 0
        test_data = []
        labels = []
        predicted_labels = []
        feature_names = []

        for sample, encoded_sample in data:
            start_time_ref = time.process_time_ns()
            feature_names = np.array(encoded_sample.features)
            prediction = self.model_instance.predict(encoded_sample)

            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    test_data.append(encoded_sample[i])
                    labels.append(s[PredictionField.GROUND_TRUTH])
                    predicted_labels.append(prediction[i])

                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_CLASS] = prediction[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                test_data.append(encoded_sample)
                labels.append(sample[PredictionField.GROUND_TRUTH])
                predicted_labels.append(prediction[0])

                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_CLASS] = prediction[0]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        self.data_set = np.array(test_data)
        self.test_data['X'] = self.data_set
        self.test_data['y'] = np.array(labels)
        self.test_data['y_predict'] = np.array(predicted_labels)
        self.feature_names = feature_names

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)

    def get_training_data(self):
        return self.train_data
        # return self.encoded_features

    def get_labels_name(self):
        return self.feature_names

    def get_test_data(self):
        return self.test_data
        # return self.data_set

    def predict_proba(self, data):
        return self.model_instance.predict_proba(data)

    def explain_with_rf(self, train_data=None, test_data=None, number_of_samples=None):
        return self.model_instance.feature_importances_

    def explain_with_trustee(self, X_train=None, y_train=None, X_test=None, y_test=None, y_predict=None, number_of_samples=None):
        if X_train is None or y_train is None:
            log.error("Training data is not provided")
            return

        trust_report = TrustReport(
            self.model_instance,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            max_iter=1,
            num_pruning_iter=2,
            trustee_num_iter=30,
            trustee_num_stability_iter=20,
            trustee_sample_size=0.3,
            analyze_branches=True,
            analyze_stability=True,
            top_k=10,
            verbose=False,
            skip_retrain=True,
            class_names=list(self.existing_labels.values()),
            feature_names=self.feature_names,
            is_classify=True,
        )

        save_path = os.path.dirname(self.store_file)
        print(trust_report.trustee.get_top_features())

        dt_representation = tree_to_bracket_notation(trust_report.max_dt, self.feature_names, recurse_tree_with_classes)
        pruned_dt_representation = tree_to_bracket_notation(trust_report.min_dt, self.feature_names,
                                                            recurse_tree_with_classes)

        with open(f"{save_path}/dt_representation.txt", "w") as f:
            f.write(dt_representation)

        with open(f"{save_path}/pruned_dt_representation.txt", "w") as f:
            f.write(pruned_dt_representation)

        dump(trust_report.max_dt, f"{save_path}/dt.pickle")
        dump(trust_report.min_dt, f"{save_path}/pruned_dt.pickle")

        dump({"X": X_test, "y": y_test, "y_predict": y_predict}, f"{save_path}/{self.model_name}_test-data.pickle")
        trust_report.save(save_path)