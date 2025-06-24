import time
from typing import Generator, Any, Dict, Tuple, Optional

import lime
import numpy
import numpy as np
import shap
from joblib import dump, load
from lime import lime_tabular
from scipy.special import erf
from sklearn.ensemble import IsolationForest

# from trustee.report.trust import TrustReport
from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from common.pipeline_logger import PipelineLogger
from models.trustee.IAnomalyDetectionModelXAI import IAnomalyDetectionModelXAI
from xai.ITrusteeExplainableModel import ITrusteeExplainableModel

log = PipelineLogger.get_logger()


class IsolationForestModelXAI(IAnomalyDetectionModelXAI, ITrusteeExplainableModel):
    def __init__(
            self,
            model_name,
            train_new_model=True,
            skip_saving_model=False,
            model_storage_base_path=None,
            model_relative_path=None,
            filter_label: Optional[int] = None,
            train_label: Optional[int] = None,
            **kwargs,
    ):
        self.model_instance = None
        self.filter_label = filter_label
        self.feature_names = []
        self.train_label = train_label

        self.train_data = {}
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
        log.info("Training an IsolationForest model.")
        data_prep_time = 0

        # single_array_processing = False
        # concatenated_data_array = None
        encoded_features = []
        labels = []
        feature_names = []

        for samples, encoding in data:
            if not feature_names:
                feature_names = np.array(encoding.features)

            start = time.process_time_ns()

            if isinstance(samples, list):
                # if self.filter_label:
                #     # TODO filter xarray by GROUND_TRUTH filter.
                #     pass
                # elif concatenated_data_array is None:
                #     concatenated_data_array = encoding
                # else:
                #     concatenated_data_array = numpy.concatenate(
                #         (concatenated_data_array, encoding),
                #         axis=0,
                #     )
                for s in samples:
                    labels.append(s[PredictionField.GROUND_TRUTH])

                if len(encoded_features) == 0:
                    encoded_features = encoding
                else:
                    encoded_features = numpy.concatenate(
                        (encoded_features, encoding), axis=0
                    )
            else:
                # single_array_processing = True
                labels.append(samples[PredictionField.GROUND_TRUTH])
                encoded_features.append(encoding[0])
            data_prep_time += time.process_time_ns() - start

        if self.train_label is not None:
            train_encoded_features = [encoding for encoding, label in zip(encoded_features, labels) if label == self.train_label]
        else:
            train_encoded_features = encoded_features

        training_start = time.process_time_ns()
        # TODO make model parameters configurable.
        self.model_instance = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',
                                              max_features=1.0)
        self.model_instance.fit(train_encoded_features)

        # if not single_array_processing:
        #     self.model_instance.fit(concatenated_data_array)
        # else:
        #     self.model_instance.fit(encoded_features)
        training_time = time.process_time_ns() - training_start

        self.train_data['X'] = np.array(encoded_features)
        self.train_data['y'] = np.array(labels)
        self.train_data['feature_names'] = feature_names

        sample_count = len(encoded_features)

        report_performance(type(self).__name__ + "-preparation", log, sample_count,
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, sample_count,
                           training_time)

        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

    def load(self):
        self.model_instance = load(self.store_file)

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
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
            score = self.model_instance.decision_function(encoded_sample)
            prediction = self.model_instance.predict(encoded_sample)

            if isinstance(sample, list):
                # Handle the prediction for multi-sample encoding.
                for i, s in enumerate(sample):
                    test_data.append(encoded_sample[i])
                    labels.append(s[PredictionField.GROUND_TRUTH])
                    predicted_labels.append(prediction[i] == -1)

                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i] == -1
                    s[PredictionField.OUTPUT_DISTANCE] = score[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s

            else:
                test_data.append(encoded_sample)
                labels.append(sample[PredictionField.GROUND_TRUTH])
                predicted_labels.append(prediction[0] == -1)

                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction[0] == -1
                sample[PredictionField.OUTPUT_DISTANCE] = score[0]
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        self.test_data['X'] = np.array(test_data)
        self.test_data['y'] = np.array(labels)
        self.test_data['y_predict'] = np.array(predicted_labels)
        self.test_data['feature_names'] = feature_names
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

    def predict_proba(self, data, **kwargs):
        return self.model_instance.predict_proba(data)

    def explain_with_shap(self, train_data=None, test_data=None, number_of_samples=None, **kwargs):
        if test_data is None:
            log.error("No testing data is provided")
            return

        shap_values = shap.TreeExplainer(self.model_instance).shap_values(test_data)
        print("SHAP VALUES: ", shap_values.shape)
        # the original shape of shap_values is (number_of_samples, number_of_features, 2)
        # after transpose(2, 0, 1), its shape is (2, number_of_samples, number_of_features)
        # shap_values = shap_values.transpose(2, 0, 1)
        # shap.summary_plot(list(shap_values), data, feature_names=self.feature_names)
        # then, get absolute of shap_values[1], which is (number_of_samples, number_of_features)
        # and calculate mean of importance of each feature --> final result has shape of (number_of_features)
        vals = np.abs(shap_values).mean(0)
        return vals

    def convert_probabilities(self, data):
        decision_scores = self.model_instance.decision_function(data)
        probs = np.zeros([data.shape[0], 2])
        pre_erf_score = (decision_scores - np.mean(decision_scores)) / (np.std(decision_scores) * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        probs[:, 1] = erf_score.clip(0, 1).ravel()
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    def explain_with_lime(self, train_data=None, test_data=None, number_of_samples=None, feature_names=None, **kwargs):
        if train_data is None or test_data is None:
            log.error("Training data or Testing data is not provided")
            return

        explainer = lime.lime_tabular.LimeTabularExplainer(train_data, mode='classification', class_names=['0', '1'],
                                                           feature_names=feature_names, verbose=False)

        if not number_of_samples:
            number_of_samples = len(test_data)
        elif number_of_samples <= 0:
            log.info(f"Number of samples for LIME interpretation is set to {number_of_samples}, which is invalid. Therefore, numbe of samples for interpreration is re-assigned to number of samples in provided dataset")
            number_of_samples = len(test_data)

        explain_values = np.empty((number_of_samples, len(feature_names)))
        for i in range(test_data[:number_of_samples, :].shape[0]):
            exp = explainer.explain_instance(test_data[i], self.convert_probabilities,
                                             num_features=len(feature_names))
            exp_map = exp.as_map()
            feat = [exp_map[1][m][0] for m in range(len(exp_map[1]))]
            weight = [exp_map[1][m][1] for m in range(len(exp_map[1]))]
            mapping = dict(zip(feat, weight))
            sorted_dict = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[0])}
            explain_values[i] = list(sorted_dict.values())

        vals = np.abs(explain_values).mean(0)
        return vals

    def predict_for_trustee(self, data):
        raw_predictions = self.model_instance.predict(data)
        predictions = [raw_prediction == -1 for raw_prediction in raw_predictions]
        return np.array(predictions)

    def get_model_for_trustee(self):
        return self

    def get_prediction_method_name_for_trustee(self):
        return "predict_for_trustee"
