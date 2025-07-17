import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

# import numpy as np
# np.float = float
# np.int = np.int32
# np.bool = np.bool_

from river import forest
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


class AdaptativeRandomForestModelXAI(IAnomalyDetectionModelXAI, ITrusteeExplainableModel):
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
        **kwargs,
    ):

        self.model_instance = forest.ARFClassifier(seed=8, leaf_prediction="mc", grace_period=50, n_models=10, tau=0.05)
        
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
        data: Generator[Tuple[Dict[IFeature, Any], List[float]], None, None],
        **kwargs,
        ):
        
        log.info("Training method for stream-data supervised model Adaptative Random Forest Classifier.")

        data_prep_time = 0

        single_array_processing = False
        concatenated_data_array = None

        labels = []
        encoded_features = []
        self.sample_count = 0

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

            self.sample_count += 1
            if self.sample_count % self.save_interval == 0:
                self._save_model()

        
        training_time = time.process_time_ns() - training_start

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

        sum_processing_time = 0
        sum_samples = 0
        self.sample_count = 0
        
        labels = []
        encoded_features = []
        self.sample_count = 0
        
        i = 0

        for sample, encoded_sample in data:
            i = i+1
            start_time_ref = time.process_time_ns()

            # Necessary to scale samples, but River only works with dictionaries, so transforming
            feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                'feature11', 'feature12']
            
            encoded_sample_dict = [dict(zip(feature_names, arr)) for arr in encoded_sample]

            y = sample[PredictionField.GROUND_TRUTH]

            for encoded_sample in encoded_sample_dict:
                prediction = self.model_instance.predict_one(encoded_sample)
                self.model_instance.learn_one(encoded_sample, y) # After scaling, learn

        # for sample, encoded_sample in data:
        #     i = i+1
        #     start_time_ref = time.process_time_ns()
        #     print(f"Enc sample {encoded_sample}")

        #     if isinstance(sample, list):
        #         # Handle the list with multiple samples used together with
        #         # xarray DataArray encodings.
        #         for f in sample:
        #             labels.append(f[PredictionField.GROUND_TRUTH])
        #         if not encoded_features:
        #             encoded_features = [list(encoded_sample)]  # Convert to a list of lists
        #         else:
        #             # Instead of concatenating, extend the list directly
        #             encoded_features.extend(encoded_sample)
        #     else:
        #         labels = sample[PredictionField.GROUND_TRUTH]
        #         encoded_features = list(encoded_sample[0])  # Ensure this is a list

        #     print(f"Enc feat {encoded_features}")
                
            
        #     # print(f"Labels {labels}")
        #     # print(f"Enc feat {encoded_features}")
            
        #     # Necessary to scale samples, but River only works with dictionaries, so transforming
            
        #     feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
        #         'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
        #         'feature11', 'feature12']

        #     encoded_features = [dict(zip(feature_names, arr)) for arr in encoded_features]

        #     # riverdataset = stream.iter_array(encoded_features, labels, feature_names=['x1', 'x2', 'x3', 'x4'])

        #     for x, y in zip(encoded_features, labels):
        #         print(x, y)
        #         # self.scaler.learn_one(x)
        #         # x = self.scaler.transform_one(x)  # Scale the features
        #         prediction = self.model_instance.predict_one(x)
        #         self.model_instance.learn_one(x, y) # After scaling, learn


                self.sample_count += 1
                if self.sample_count % self.save_interval == 0 and not self.skip_saving_model:
                    self._save_model()

                if i<25:
                    print(f"Prediction is {prediction}")

            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i]
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample
        
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
