import time
from typing import Any, Dict, Generator, Optional, Tuple

import numpy
import numpy as np
import shap
from joblib import dump, load
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import MeanSquaredError

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from common.pipeline_logger import PipelineLogger
from models.trustee.IAnomalyDetectionModelXAI import IAnomalyDetectionModelXAI
from xai.ITrusteeExplainableModel import ITrusteeExplainableModel

log = PipelineLogger.get_logger()


class MLPAutoEncoderModelXAI(IAnomalyDetectionModelXAI, ITrusteeExplainableModel):
    """
    Multi-layer perceptron (MLP) based autoencoder.
    """

    def __init__(
            self,
            filter_label: Optional[int] = None,
            train_label: Optional[int] = None,
            **kwargs,
    ):
        """
        TODO: multi-AE setup where PredictionField.GROUND_TRUTH label of the data
         will be used to split the data into multiple training sets, then training
         multiple AEs to each predict their own class only.

        :param filter_label:
        :param kwargs: Arguments for the superclass constructor.
        """
        self.model_instance = None
        self.filter_label = filter_label
        self.threshold = None
        # self.encoded_features = []
        # self.data_set = []
        self.feature_names = []
        self.scaler = StandardScaler()

        self.train_label = train_label
        self.train_data = {}
        self.test_data = {}

        super().__init__(**kwargs)

    def get_reconstruction_error(self, x):
        r = self.model_instance.predict(x)
        return MeanSquaredError().call(x, r)

    def train(
            self,
            data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None],
            **kwargs,
    ):
        log.info("Training an MLP autoencoder.")
        data_prep_time = 0

        # single_array_processing = False
        # concatenated_data_array = None
        encoded_features = []
        labels = []
        feature_names = []

        for samples, encoding in data:
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

                if not feature_names:
                    feature_names = np.array(encoding.features)

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

        self.scaler.fit(encoded_features)
        scaled_train_encoded_features = self.scaler.transform(np.array(train_encoded_features))

        training_start = time.process_time_ns()
        # TODO make model parameters configurable.
        self.model_instance = MLPRegressor(
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

        self.model_instance.fit(scaled_train_encoded_features, scaled_train_encoded_features)
        train_loss = self.get_reconstruction_error(scaled_train_encoded_features)

        # if not single_array_processing:
        #     self.model_instance.fit(concatenated_data_array, concatenated_data_array)
        #     train_loss = self.get_reconstruction_error(concatenated_data_array)
        # else:
        #     self.model_instance.fit(encoded_features, encoded_features)
        #     train_loss = self.get_reconstruction_error(encoded_features)
        training_time = time.process_time_ns() - training_start

        self.train_data['X'] = np.array(encoded_features)
        self.train_data['scaled_X'] = scaled_train_encoded_features
        self.train_data['y'] = np.array(labels)
        self.train_data['feature_names'] = feature_names
        self.threshold = np.mean(train_loss.numpy()) + np.std(train_loss.numpy())

        sample_count = len(encoded_features)

        report_performance(type(self).__name__ + "-preparation", log, sample_count,
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, sample_count,
                           training_time)

        if not self.skip_saving_model:
            dump({"model": self.model_instance, "threshold": self.threshold}, self.store_file)
            dump(self.scaler, f'{self.store_file}-scaler')

    def load(self):
        model_info = load(self.store_file)
        if not model_info:
            log.error(f"Failed to load model from: {self.store_file}")
            return

        self.model_instance = model_info['model']
        self.threshold = model_info['threshold']
        if not self.model_instance or self.threshold:
            log.error(f"Failed to load model from: {self.store_file}")

        self.scaler = load(f'{self.store_file}-scaler')

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

            scaled_encoded_sample = self.scaler.transform(encoded_sample)
            prediction = self.model_instance.predict(scaled_encoded_sample)

            if isinstance(sample, list):
                # Handle the prediction for multi-sample encoding.
                for i, s in enumerate(sample):
                    predicted_class = MeanSquaredError().call(scaled_encoded_sample[i],
                                                                               prediction[i]).numpy() > self.threshold

                    test_data.append(encoded_sample[i])
                    labels.append(s[PredictionField.GROUND_TRUTH])
                    predicted_labels.append(predicted_class)

                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = predicted_class
                    s[PredictionField.OUTPUT_DISTANCE] = sum(abs(prediction[i]))
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s

            else:
                predicted_class = MeanSquaredError().call(scaled_encoded_sample,
                                                          prediction[0]).numpy() > self.threshold

                test_data.append(encoded_sample)
                labels.append(sample[PredictionField.GROUND_TRUTH])
                predicted_labels.append(predicted_class)

                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = predicted_class
                sample[PredictionField.OUTPUT_DISTANCE] = sum(prediction[0])
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        self.test_data['X'] = np.array(test_data)
        self.test_data['scaled_X'] = self.scaler.transform(self.test_data['X'])
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
        pass

    def predict_for_shap(self, data):
        prediction = self.model_instance.predict(data)
        return MeanSquaredError().call(data, prediction[0]).numpy() > self.threshold

    def explain_with_shap(self, train_data=None, test_data=None, number_of_samples=None, **kwargs):
        if train_data is None or test_data is None:
            log.error("No training data or testing data is provided")
            return

        if number_of_samples is None:
            number_of_samples = len(test_data)

        background = shap.kmeans(train_data, 10)
        e = shap.KernelExplainer(self.predict_for_shap, background)
        shap_values = e.shap_values(test_data[:number_of_samples, :])
        # print ("shap_values: ", shap_values)
        vals = np.abs(shap_values).mean(0)
        return vals

    def predict_for_trustee(self, data):
        raw_predictions = self.model_instance.predict(data)
        predictions = MeanSquaredError().call(data, raw_predictions).numpy() > self.threshold
        # predictions = [MeanSquaredError().call(sample, raw_prediction).numpy() > self.threshold for sample, raw_prediction in zip(data, raw_predictions)]
        return np.array(predictions)

    def explain_with_trustee(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        class_names=None,
        feature_names=None,
        max_iter=5,
        num_pruning_iter=2,
        trustee_num_iter=50,
        trustee_num_stability_iter=20,
        trustee_sample_size=0.3,
        top_k=10,
        save_path=None,
        **kwargs
    ):
        # print(self.feature_names)
        if X_train is None or y_train is None or X_test is None or y_test is None:
            log.error("Training data or Testing data is not provided")
            return

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        super().explain_with_trustee(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_names,
            max_iter,
            num_pruning_iter,
            trustee_num_iter,
            trustee_num_stability_iter,
            trustee_sample_size,
            top_k,
            save_path,
            **kwargs
        )

    def explain_with_mytrustee(
            self,
            X_train=None,
            y_train=None,
            X_test=None,
            y_test=None,
            class_names=None,
            feature_names=None,
            max_iter=5,
            num_pruning_iter=2,
            trustee_num_iter=50,
            trustee_num_stability_iter=20,
            trustee_sample_size=0.3,
            top_k=10,
            save_path=None,
            **kwargs
    ):
        # print(self.feature_names)
        if X_train is None or y_train is None or X_test is None or y_test is None:
            log.error("Training data or Testing data is not provided")
            return

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        super().explain_with_mytrustee(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_names,
            max_iter,
            num_pruning_iter,
            trustee_num_iter,
            trustee_num_stability_iter,
            trustee_sample_size,
            top_k,
            save_path,
            **kwargs
        )

    def get_prediction_method_name_for_trustee(self):
        return "predict_for_trustee"
