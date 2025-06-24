import logging
from abc import ABC

from joblib import dump
from trustee.report.trust import TrustReport

from utils.my_trustee.report.trust import TrustReport as MyTrustReport
from utils import recurse_tree_with_classes, tree_to_bracket_notation

log = logging.getLogger()


class ITrusteeExplainableModel(ABC):
    """
    Generic interface for SHAP explainable models to implement.
    """

    def get_model_for_trustee(self):
        return self

    def get_prediction_method_name_for_trustee(self):
        return "predict_one"

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
            prediction_method_name="predict_one",
            **kwargs
    ):
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
