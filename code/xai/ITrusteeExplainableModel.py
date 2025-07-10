from typing import List, Generator, Tuple, Dict, Any
import xarray
import numpy as np
import pandas as pd

import logging
from abc import ABC

from joblib import dump
from trustee.report.trust import TrustReport

from utils.my_trustee.report.trust import TrustReport as MyTrustReport
from utils.stream_trustee.report.trust import TrustReport as StreamTrustReport
from utils import recurse_tree_with_classes, tree_to_bracket_notation

log = logging.getLogger()


class ITrusteeExplainableModel(ABC):
    """
    Generic interface for SHAP explainable models to implement.
    """

    def get_model_for_trustee(self):
        return self

    def get_prediction_method_name_for_trustee(self):
        return "predict"

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
        if class_names is None:
            class_names = ['benign', 'anomalous']

        if X_train is None or y_train is None:
            log.error("Training data is not provided")
            return

        trust_report = TrustReport(
            self.get_model_for_trustee(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            max_iter=max_iter,
            num_pruning_iter=num_pruning_iter,
            trustee_num_iter=trustee_num_iter,
            trustee_num_stability_iter=trustee_num_stability_iter,
            trustee_sample_size=trustee_sample_size,
            analyze_branches=True,
            analyze_stability=True,
            top_k=top_k,
            verbose=False,
            skip_retrain=True,
            class_names=class_names,
            feature_names=feature_names,
            is_classify=True,
            predict_method_name=self.get_prediction_method_name_for_trustee()
        )

        # print(trust_report.trustee.get_top_features())
        if save_path:
            trust_report.save(save_path)

            dt_representation = tree_to_bracket_notation(trust_report.max_dt, feature_names, recurse_tree_with_classes)
            pruned_dt_representation = tree_to_bracket_notation(trust_report.min_dt, feature_names,
                                                                recurse_tree_with_classes)

            with open(f"{save_path}/dt_representation.txt", "w") as f:
                f.write(dt_representation)

            with open(f"{save_path}/pruned_dt_representation.txt", "w") as f:
                f.write(pruned_dt_representation)

            dump(trust_report.max_dt, f"{save_path}/dt.pickle")
            dump(trust_report.min_dt, f"{save_path}/pruned_dt.pickle")

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
            prediction_method_name="predict",
            **kwargs
    ):



        print(f"@@@ Here is the blackbox type in ITrusteeExplainableModel: {type(self.get_model_for_trustee())}")
        print(type(self))
        print(X_train)
        print(type(X_train))

        print("@@@ Methods called in 4: ITrusteeExplainableModel")
        print([m for m in dir(self.model_instance) if callable(getattr(self.model_instance, m))])

        prediction = self.model_instance.predict(X_train)      # make prediction
        print("Prediction ITrusteeExplainableModel:", prediction)

        print("@@@ Finished ITrusteeExplainableMOdel Prediction")
















        if class_names is None:
            class_names = ['benign', 'anomalous']

        if X_train is None or y_train is None:
            log.error("Training data is not provided")
            return

        trust_report = MyTrustReport(
            self.get_model_for_trustee(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            max_iter=max_iter,
            num_pruning_iter=num_pruning_iter,
            trustee_num_iter=trustee_num_iter,
            trustee_num_stability_iter=trustee_num_stability_iter,
            trustee_sample_size=trustee_sample_size,
            analyze_branches=True,
            analyze_stability=True,
            top_k=top_k,
            verbose=False,
            skip_retrain=True,
            class_names=class_names,
            feature_names=feature_names,
            is_classify=True,
            predict_method_name=prediction_method_name
        )

        # print(trust_report.trustee.get_top_features())
        if save_path:
            trust_report.save(save_path)

            dt_representation = tree_to_bracket_notation(trust_report.max_dt, feature_names, recurse_tree_with_classes)
            pruned_dt_representation = tree_to_bracket_notation(trust_report.min_dt, feature_names,
                                                                recurse_tree_with_classes)

            with open(f"{save_path}/dt_representation.txt", "w") as f:
                f.write(dt_representation)

            with open(f"{save_path}/pruned_dt_representation.txt", "w") as f:
                f.write(pruned_dt_representation)

            dump(trust_report.max_dt, f"{save_path}/dt.pickle")
            dump(trust_report.min_dt, f"{save_path}/pruned_dt.pickle")

    def explain_with_stream_trustee(
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
            prediction_method_name="predict_one",
            **kwargs
    ):


        print(f"@@@ Here is the blackbox type in ITrusteeExplainableModel: {type(self.get_model_for_trustee())}")
        print(type(self))
        print(X_train)
        print(type(X_train))

        # Convert
        river_data = self.array_to_river_dicts(X_train, feature_names)

        counts = [0, 0]

        print("@@@ Methods called in 4-A self: ITrusteeExplainableModel")
        print([m for m in dir(self) if callable(getattr(self, m))])
        print("@@@ Methods called in 4-B self.model_instance: ITrusteeExplainableModel")
        print([m for m in dir(self.model_instance) if callable(getattr(self.model_instance, m))])
        print("@@@ Methods called in 4-C many: ITrusteeExplainableModel")
        print(type(self))
        print(type(self.scaler))
        print(type(self.model_instance))

        # Loop through samples
        for sample in river_data:
            prediction = self.model_instance.predict_one(sample)      # make prediction
            # scaled = self.scaler.transform_one(sample)
            # prediction = self.model_instance.predict_one(scaled)      # make prediction
            print("Prediction ITrusteeExplainableMOdel:", prediction)
            # increment safely
            if prediction == 0:
                counts[0] += 1
            elif prediction == 1:
                counts[1] += 1
            else:
                # if you ever get something unexpected, you can decide to:
                #   * force it into one of the bins
                #   * ignore it
                #   * log an error, etc.
                print(f"⚠️  unexpected label {prediction!r}, ignoring")

        print("@@@ Finished ITrusteeExplainableMOdel Prediction")
        print(counts)




        if class_names is None:
            class_names = ['benign', 'anomalous']

        if X_train is None or y_train is None:
            log.error("Training data is not provided")
            return

        trust_report = StreamTrustReport(
            self.get_model_for_trustee(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            max_iter=max_iter,
            num_pruning_iter=num_pruning_iter,
            trustee_num_iter=trustee_num_iter,
            trustee_num_stability_iter=trustee_num_stability_iter,
            trustee_sample_size=trustee_sample_size,
            analyze_branches=True,
            analyze_stability=True,
            top_k=top_k,
            verbose=False,
            skip_retrain=True,
            class_names=class_names,
            feature_names=feature_names,
            is_classify=True,
            predict_method_name=prediction_method_name
        )

        # print(trust_report.trustee.get_top_features())
        if save_path:
            trust_report.save(save_path)

            dt_representation = tree_to_bracket_notation(trust_report.max_dt, feature_names, recurse_tree_with_classes)
            pruned_dt_representation = tree_to_bracket_notation(trust_report.min_dt, feature_names,
                                                                recurse_tree_with_classes)

            with open(f"{save_path}/dt_representation.txt", "w") as f:
                f.write(dt_representation)

            with open(f"{save_path}/pruned_dt_representation.txt", "w") as f:
                f.write(pruned_dt_representation)

            dump(trust_report.max_dt, f"{save_path}/dt.pickle")
            dump(trust_report.min_dt, f"{save_path}/pruned_dt.pickle")



    def dataframe_to_river_dicts(self, df: pd.DataFrame, feature_names: list[str]) -> list[dict]:
        """
        Converts a pandas DataFrame to a list of dictionaries with specified feature names.
        Each dictionary represents one sample and can be used in River.
        """
        return [dict(zip(feature_names, row)) for row in df.values]

    def array_to_river_dicts(self,
        X: np.ndarray, 
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Converts a 2D NumPy array X of shape (n_samples, n_features)
        into a list of dicts [{feat_name: value, …}, …] for River.
        
        Parameters
        ----------
        X : np.ndarray
            Your data array, shape (n_samples, n_features).
        feature_names : List[str]
            List of length n_features giving the name for each column.
        
        Returns
        -------
        List[Dict[str, Any]]
            One dict per row in X, mapping feature_names[i] -> X[row, i].
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if len(feature_names) != n_features:
            raise ValueError(
                f"Number of feature_names ({len(feature_names)}) "
                f"does not match number of columns in X ({n_features})"
            )
        return [
            dict(zip(feature_names, X[i]))
            for i in range(n_samples)
        ]