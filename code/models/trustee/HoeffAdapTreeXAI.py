import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

import numpy as np
np.float = float
np.int = np.int32
np.bool = np.bool_

from river import forest, tree
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
from river import stream
from river import evaluate
from river.compose import Pipeline

from datetime import datetime
from joblib import dump, load

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.trustee.IAnomalyDetectionModelXAI import IAnomalyDetectionModelXAI
from utils import recurse_tree_with_classes, tree_to_bracket_notation
from utils.my_trustee.report.trust import TrustReport as MyTrustReport
from xai.ITrusteeExplainableModel import ITrusteeExplainableModel

from common.pipeline_logger import PipelineLogger
log = PipelineLogger.get_logger()

import matplotlib.pyplot as plt
import pickle

from enum import Enum


class HoeffdingAdaptativeTreeModelXAI(IAnomalyDetectionModelXAI, ITrusteeExplainableModel):
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
        save_interval=500,  # Interval to save model periodically
        # grace_period=0,     # Default value
        # delta=1e-5,        # Default value
        # leaf_prediction="nb",  # Default value
        # nb_threshold=0,     # Default value
        # seed=0,             # Default value
        # tau=0.05,           # Default value
        **kwargs,
    ):
        
        self.model_instance = tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, delta=1e-5, leaf_prediction='nb', nb_threshold=0, seed=0, tau=0.05, switch_significance=0.05, binary_split=False, min_branch_fraction=0.01)
        # self.model_instance = tree.HoeffdingAdaptiveTreeClassifier(grace_period=self.grace_period, delta=self.delta, leaf_prediction=self.leaf_prediction, nb_threshold=self.nb_threshold, seed=self.seed, tau=self.tau)        
        # self.model_instance = tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, delta=1e-5, leaf_prediction='nb', nb_threshold=10, seed=0)

        self.scaler = preprocessing.StandardScaler()
        # self.scaler = preprocessing.MinMaxScaler()
        # self.scaler = preprocessing.AdaptativeStandardScaler(fading_factor=.3)
        # self.scaler = preprocessing.RobustScaler()

        self.model_instance = Pipeline(
            self.scaler,
            self.model_instance
            )
        
        self.last_scores = []
        self.save_interval = save_interval
        self.sample_count = 0

        self.encoded_features = []
        self.train_data = {}
        self.feature_names = []
        self.data_set = []
        self.test_data = {}

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
        # data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None],
        data: Generator[Tuple[Dict[IFeature, Any], List[float]], None, None],
        **kwargs,
        ):
        
        log.info("Training method for stream-data supervised model Hoeffding Adaptative Tree Classifier.")

        data_prep_time = 0
        sum_processing_time = 0
        label = []
        labels = []
        feature_sample = []
        feature_array = []
        feature_names = []
        feature_dict = {}

        data_prep_time = 0
        start = time.process_time_ns()
        training_start = time.process_time_ns()
        for samples, encoding in data:
            if len(feature_names) == 0:
                feature_names = list(encoding.features.values)
            if isinstance(samples, list):
                # River don't allow for batch training/testing
                raise TypeError("Streaming algorithms cannot handle batches of samples.")
            else:
                feature_sample = encoding[0].values
                feature_dict = dict(zip(feature_names, feature_sample))
                label = samples[PredictionField.GROUND_TRUTH]
                self.model_instance.learn_one(feature_dict, label) # After scaling, learn

                self.sample_count += 1
                if self.sample_count % self.save_interval == 0:
                    self._save_model()
                feature_array.append(feature_sample)
                labels.append(label)

                
        data_prep_time += time.process_time_ns() - start
        training_time = time.process_time_ns() - training_start

        self.train_data['X'] = feature_array
        self.train_data['y'] = labels
        self.train_data['feature_names'] = feature_names

        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)
        if not self.skip_saving_model:
            dump(self.model_instance, self.store_file)

    def _save_model(self):
        dump(self.model_instance, self.store_file)
        log.info(f"Model saved to {self.store_file}")

    def load(self, **kwargs):
        self.model_instance = load(self.store_file)
        if not self.model_instance:
            log.error(f"Failed to load model from: {self.store_file}")

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        data_prep_time = 0
        sum_processing_time = 0
        sum_samples = 0
        label = []
        labels = []
        feature_sample = []
        feature_array = []
        feature_names = []
        feature_dict = {}
        predicted_labels = []

        for sample, encoding in data:
            if len(feature_names) == 0:
                feature_names = list(encoding.features.values)

            start_time_ref = time.process_time_ns()
            if isinstance(sample, list):
                # River don't allow for batch training/testing
                raise TypeError("Streaming algorithms cannot handle batches of samples.")
            else:
                feature_sample = encoding[0].values
                feature_dict = dict(zip(feature_names, feature_sample))
                label = sample[PredictionField.GROUND_TRUTH]
                prediction = self.model_instance.predict_one(feature_dict)
                feature_array.append(feature_sample)
                labels.append(label)
                predicted_labels.append(prediction)

                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        self.data_set = np.array(feature_array)
        print("Feature array type:", type(self.data_set))
        print("Feature array[0] shape:", feature_array[0].shape)
        print("self.data_set shape:", self.data_set.shape)
        print("self.data_set dtype:", self.data_set.dtype)
        self.test_data['X'] = self.data_set
        self.test_data['y'] = labels
        self.test_data['y_predict'] = predicted_labels
        self.test_data['feature_names'] = feature_names

        self.feature_names = feature_names

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)


   
    def evaluate(self, data: Generator, **kwargs) -> Tuple[list[int], list[float]]:
        
        metric = metrics.Accuracy()

        y_pred = []
        cumulative_accuracies = []
        
        for x, y in data:

            y_p = self.model_instance.predict_one(x)
            y_pred.append(y_p)
            metric.update(y, y_p)
            cumulative_accuracies.append(metric.get() * 100)
            self.model_instance.learn_one(x, y)

        # steps = evaluate.iter_progressive_val_score(
        #             dataset=data,
        #             model=self.model_instance,
        #             metric=metrics.Accuracy(),
        #         )

        # cumulative_accuracies = []

        # for step in steps:
        #     accuracy_value = step['Accuracy'].get() * 100  # Get the accuracy as a percentage
        #     cumulative_accuracies.append(accuracy_value)  # Append to the list
        #     # print(accuracy_value)
        
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
