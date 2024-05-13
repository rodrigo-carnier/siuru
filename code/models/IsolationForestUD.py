### LAURA'S RF MODEL IMPORTS

import logging
import time
from typing import Generator, Any, Dict, Tuple

import numpy
import numpy as np
from joblib import dump, load

# from sklearn.ensemble import RandomForestClassifier

from common.features import EncodedSampleGenerator, IFeature, PredictionField, SampleGenerator
from common.functions import report_performance
from models.IAnomalyDetectionModel import IAnomalyDetectionModel

log = logging.getLogger()



### UYEN'S RF MODEL IMPORTS

from sklearn.ensemble import IsolationForest

# import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score,accuracy_score,mean_squared_error, log_loss, confusion_matrix
# import shap
import pandas as pd
## import numpy as np
# from tensorflow.keras.models import Model

# Lime imports
import lime
import lime.lime_tabular

## import logging
import os.path
from typing import Generator, Any, Optional, List, Dict
## import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
## log = logging.getLogger()
from scipy.special import erf



# class RandomForestModel(IAnomalyDetectionModel):
#     def __init__(
#         self,
#         model_name,
#         train_new_model=True,
#         skip_saving_model=False,
#         model_storage_base_path=None,
#         model_relative_path=None,
#         **kwargs,
#     ):
#         self.model_instance = None
#         super().__init__(
#             model_name,
#             train_new_model=train_new_model,
#             skip_saving_model=skip_saving_model,
#             model_storage_base_path=model_storage_base_path,
#             model_relative_path=model_relative_path,
#             **kwargs,
#         )

class IsolationForestModel(IsolationForest):
    def __init__(
        self,
        features,
        labels,
        feature_names
    ):
        X_train,
        X_test,
        y_train,
        y_test = train_test_split(
            features, labels, test_size=0.33, random_state=8
        )
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.feature_names = feature_names
       
        self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)

    def convert_probabilities(self, data):
        decision_scores = self.model.decision_function(data)
        probs = np.zeros([data.shape[0], 2])
        pre_erf_score = ( decision_scores - np.mean(decision_scores) ) / ( np.std(decision_scores) * np.sqrt(2) )
        erf_score = erf(pre_erf_score)
        probs[:, 1] = erf_score.clip(0, 1).ravel()
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    def train(self):
        X_train_normal = [f for i, f in enumerate(self.X_train) if self.y_train[i] == 0]
        self.model.fit(X_train_normal)
    def custom_predict(self):
        y_pred = self.model.predict(self.X_test)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        y_test = self.y_test
        performance = [accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='macro')]
        return performance
    # def explain_with_shap(self):
    #     X_explain = self.X_test
    #     shap_values = shap.TreeExplainer(self.model).shap_values(X_explain)
    #     vals = np.abs(shap_values).mean(0)
    #     # shap.summary_plot(shap_values, features = X_explain, feature_names = feature_names)
    #     # shap.summary_plot(shap_values, X_explain,feature_names = self.feature_names, plot_type = "bar")
    #     return vals
    # def explain_with_lime(self):
    #     # feature_name = ["#" + str(i) for i in range(1, 26)]
    #     explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train, mode='classification',class_names=['0', '1'], feature_names = self.feature_names, verbose=False )

    #     explain_values =np.empty((10, 25))
    #     for i in range(self.X_test[:10,:].shape[0]):
    #         exp = explainer.explain_instance(self.X_test[i], self.convert_probabilities, num_features=25)
    #         exp_map = exp.as_map()
    #         feat = [exp_map[1][m][0] for m in range(len(exp_map[1]))]
    #         weight = [exp_map[1][m][1] for m in range(len(exp_map[1]))]
    #         mapping = dict(zip(feat, weight))
    #         sorted_dict = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[0])}
    #         explain_values[i] = list(sorted_dict.values())

    #     vals = np.abs(explain_values).mean(0)
    #     return vals