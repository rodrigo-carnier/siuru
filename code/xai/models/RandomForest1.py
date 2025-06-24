import logging
import os
import time
from typing import Generator, Any, Dict, Tuple

import lime
import numpy
import numpy as np
import shap
from joblib import dump, load
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from utils import recurse_tree_with_classes, tree_to_bracket_notation
from utils.my_trustee.report.trust import TrustReport as MyTrustReport
from xai.ITrusteeExplainableModel import ITrusteeExplainableModel

log = logging.getLogger()


class RandomForestModel(IAnomalyDetectionModel, ITrusteeExplainableModel):
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
        log.info("Training a random forest classifier.")

        labels = []
        encoded_features = []
        feature_names = []

        data_prep_time = 0
        for samples, encoding in data:
            if not feature_names:
                feature_names = np.array(encoding.features)

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
        self.train_data['feature_names'] = feature_names

        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

    def load(self):
        self.model_instance = load(self.store_file)

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
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
        test_data = []
        labels = []
        predicted_labels = []
        feature_names = []

        for sample, encoded_sample in data:
            if not feature_names:
                feature_names = np.array(encoded_sample.features)

            start_time_ref = time.process_time_ns()
            prediction = self.model_instance.predict(encoded_sample)
            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    test_data.append(encoded_sample[i])
                    labels.append(s[PredictionField.GROUND_TRUTH])
                    predicted_labels.append(prediction[i])

                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                test_data.append(encoded_sample)
                labels.append(sample[PredictionField.GROUND_TRUTH])
                predicted_labels.append(prediction[0])

                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction[0]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        self.data_set = np.array(test_data)
        self.test_data['X'] = self.data_set
        self.test_data['y'] = np.array(labels)
        self.test_data['y_predict'] = np.array(predicted_labels)
        self.test_data['feature_names'] = feature_names

        self.feature_names = feature_names

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)

    def get_training_data(self):
        # return self.encoded_features
        return self.train_data

    def get_labels_name(self):
        return self.feature_names

    def get_test_data(self):
        return self.test_data
        # return self.data_set

    def predict_proba(self, data, **kwargs):
        return self.model_instance.predict_proba(data)

    def explain_with_rf(self, train_data=None, test_data=None, **kwargs):
        # the data argument is for compatible purpose
        return self.model_instance.feature_importances_

    def explain_with_shap(self, train_data=None, test_data=None, number_of_samples=None, **kwargs):
        if test_data is None:
            log.error("No training data is provided")
            return

        shap_values = shap.TreeExplainer(self.model_instance).shap_values(test_data)
        print("SHAP VALUES: ", shap_values.shape)

        # the original shape of shap_values is (number_of_samples, number_of_features, 2)
        # after transpose(2, 0, 1), its shape is (2, number_of_samples, number_of_features)
        shap_values = shap_values.transpose(2, 0, 1)
        # shap.summary_plot(list(shap_values), data, feature_names=self.feature_names)
        # then, get absolute of shap_values[1], which is (number_of_samples, number_of_features)
        # and calculate mean of importance of each feature --> final result has shape of (number_of_features)
        vals = np.abs(shap_values[1]).mean(0)
        return vals

    def explain_with_lime(self, train_data=None, test_data=None, number_of_samples=None, feature_names=None, **kwargs):
        if train_data is None or test_data is None or feature_names is None:
            log.error("Training data or Testing data or Feature names is not provided")
            return

        explainer = lime.lime_tabular.LimeTabularExplainer(train_data, mode='classification', class_names=['0', '1'],
                                                           feature_names=feature_names, verbose=False)

        if not number_of_samples:
            number_of_samples = len(test_data)
        elif number_of_samples <= 0:
            log.info(
                f"Number of samples for LIME interpretation is set to {number_of_samples}, which is invalid. Therefore, numbe of samples for interpreration is re-assigned to number of samples in provided dataset")
            number_of_samples = len(test_data)

        explain_values = np.empty((number_of_samples, len(feature_names)))
        for i in range(test_data[:number_of_samples, :].shape[0]):
            exp = explainer.explain_instance(test_data[i], self.model_instance.predict_proba,
                                             num_features=len(feature_names))
            exp_map = exp.as_map()
            feat = [exp_map[1][m][0] for m in range(len(exp_map[1]))]
            weight = [exp_map[1][m][1] for m in range(len(exp_map[1]))]
            mapping = dict(zip(feat, weight))
            sorted_dict = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[0])}
            explain_values[i] = list(sorted_dict.values())

        vals = np.abs(explain_values).mean(0)
        return vals

    def get_model_for_trustee(self):
        return self.model_instance
