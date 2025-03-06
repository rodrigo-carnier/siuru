import time
from typing import Any, Dict, Generator, Optional, List, Tuple, Union

# import numpy
import numpy as np
np.float = float
np.int = np.int32
np.bool = np.bool_

from river import naive_bayes
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
from river import stream
from river import evaluate
from datetime import datetime

from joblib import dump, load

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


import matplotlib.pyplot as plt
import pickle

from enum import Enum


class SK2RiverMLPModel(IAnomalyDetectionModel):
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

        self.model_instance = naive_bayes.GaussianNB()

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
        

        self.scaler = preprocessing.StandardScaler()
        # self.scaler = preprocessing.MinMaxScaler()
        # self.scaler = preprocessing.AdaptativeStandardScaler(fading_factor=.3)
        # self.scaler = preprocessing.RobustScaler()

        
        self.last_scores = []
        self.save_interval = save_interval
        self.sample_count = 0
        self.grace_period = 100;


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
        
        log.info("Training method for stream-data supervised model Hoeffding Adaptative Tree Classifier.")

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
        
        self.anomaly_threshold = 0.2
        # self.anomaly_threshold = 0.873

        i=0
        for sample, encoded_sample in data:
            i = i+1
            start_time_ref = time.process_time_ns()

            print(f"sample {encoded_sample}")
            # Necessary to scale samples, but River only works with dictionaries, so transforming
            feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                'feature11', 'feature12']
            

            encoded_sample = [dict(zip(feature_names, arr)) for arr in encoded_sample]
            print(f"sample dict {encoded_sample}")

            for x in encoded_sample:
                print(f"x {x}")
                prediction = self.model_instance.predict_one(x)

                self.sample_count += 1
                if self.sample_count % self.save_interval == 0 and not self.skip_saving_model:
                    self._save_model()

            if i<25:
                print(prediction)

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