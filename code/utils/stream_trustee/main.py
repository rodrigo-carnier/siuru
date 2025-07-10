
"""
Trustee
====================================
The core module of the Trustee project
"""
from typing import List, Generator, Tuple, Dict, Any
import xarray

import abc
import functools
import numpy as np
import pandas as pd

from copy import deepcopy
from collections import Counter

import apted
from apted.helpers import Tree

from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from trustee.utils.tree import get_dt_info, top_k_prune
from trustee.utils.dataset import convert_to_df, convert_to_series
from .utils import tree_representation

def dataframe_to_river_dicts(df: pd.DataFrame, feature_names: list[str]) -> list[dict]:
    """
    Converts a pandas DataFrame to a list of dictionaries with specified feature names.
    Each dictionary represents one sample and can be used in River.
    """
    return [dict(zip(feature_names, row)) for row in df.values]

def reconstruct_encoded_generator(
    test_data: Dict[str, Any]
) -> Generator[Tuple[Dict[str, Any], xarray.DataArray], None, None]:
    """
    Convert test_data dict back to an EncodedSampleGenerator.
    """

    X = test_data["X"]
    y = test_data["y"]
    feature_names = test_data["feature_names"]

    for i in range(len(X)):
        # Simulate the Sample (dict containing at least ground truth)
        sample = {PredictionField.GROUND_TRUTH: y[i]}

        # Simulate the encoded xarray.DataArray with 2D shape
        encoded = xarray.DataArray(
            [X[i]],  # Make it 2D
            dims=["samples", "features"],
            coords={"features": feature_names}
        )

        yield (sample, encoded)

def _check_if_trained(func):
    """
    Checks whether the Trustee is already fitted and self._best_student exists

    Parameters
    ----------
    func: callable
        Function to apply decorator to.
    *args: tuple
        Additional arguments should be passed as keyword arguments to `func`.
    **kwargs: dict, optional
        Extra arguments to `func`: refer to each func documentation for a list of all possible arguments.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(self._top_students) == 0:
            raise ValueError("No student models have been trained yet. Please fit() Trustee explainer first.")
        return func(self, *args, **kwargs)

    return wrapper


class Trustee(abc.ABC):
    """
    Base implementation the Trust-oriented Decision Tree Extraction (Trustee)
    algorithm to train student model based on observations from an Expert model.
    """

    def __init__(self, expert, student_class, logger=None):
        """
        Trustee constructor.

        Parameters
        ----------
        expert: object
            The ML blackbox model to analyze. The expert model must have a `predict` method call implemented for
            Trustee to work properly, unless explicitly stated otherwise using the `predict_method_name` argument
            in the fit() method.

        student_class: Class
            Class of student to train based on blackbox model predictions. The given Class must implement a `fit()
            and a `predict()` method interface for Trustee to work properly. The current implementation has been
            tested using the DecisionTreeClassifier and DecisionTreeRegressor from scikit-learn.

        logger: Logger object , default=None
            A logger object to log messages to. If none is given, the print() method will be used to log messages.
        """
        self.log = logger.log if logger else print
        self.expert = expert
        self.student_class = student_class

        self._students_by_iter = []
        self._top_students = []
        self._stable_students = []
        self._best_students = []

        self._X_train = []
        self._X_test = []
        self._y_train = []
        self._y_test = []

        self._best_student = None
        self._features = None
        self._nodes = None
        self._branches = None

        self._student_use_features: np.array = []

    @abc.abstractmethod
    def _score(self, y_true, y_pred):
        """
        Score function for student models. Compares the ground-truth predictions
        of a blackbox model with the predictions of a student model.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The ground-truth target values (class labels in classification, real numbers in regression).

        y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values (class labels in classification, real numbers in regression).

        Returns
        -------
        score: float
            Calculated student model score.
        """

    def fit(
            self,
            X,
            y,
            top_k=10,
            max_leaf_nodes=None,
            max_depth=None,
            ccp_alpha=0.0,
            train_size=0.7,
            num_iter=50,
            num_stability_iter=5,
            num_samples=2000,
            samples_size=None,
            use_features=None,
            predict_method_name="predict_one",
            optimization="fidelity",  # for comparative purposes only
            aggregate=True,  # for comparative purposes only
            verbose=False,
            feature_names: List[str] = ...
    ):
        """
        Trains Decision Tree Regressor to imitate Expert model.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to a pandas DataFrame.

        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values for X (class labels in classification, real numbers in regression).
            Internally, it will be converted to a pandas Series.

        top_k: int, default=10
            Number of top-k branches, sorted by number of samples per branch, to keep after finding
            decision tree with highest fidelity.

        max_leaf_nodes: int, default=None
            Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as
            relative reduction in impurity. If None then unlimited number of leaf nodes.

        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.

        ccp_alpha: float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the
            largest cost complexity that is smaller than ccp_alpha will be chosen. By default,
            no pruning is performed. See Minimal Cost-Complexity Pruning here for details:
            https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning

        train_size: float or int, default=0.7
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the train split. If int, represents the absolute number of train samples.

        num_iter: int, default=50
            Number of iterations to repeat Trustee inner-loop for.

        num_stability_iter: int, default=5
            Number of stability to repeat Trustee stabilization outer-loop for.

        num_samples: int, default=2000
            The absolute number of samples to fetch from the training dataset split to train the
            student decision tree model. If the `samples_size` argument is provided, this arg is
            ignored.

        samples_size: float, default=None
            The fraction of the training dataset to use to train the student decision tree model.
            If None, the value is automatically set to the `num_samples` provided value.

        use_features: array-like, default=None
            Array-like of integers representing the indexes of features from the `X` training samples.
            If not None, only the features indicated by the provided indexes will be used to train the
            student decision tree model.

        predict_method_name: str, default="predict"
            The method interface to use to get predictions from the expert model.
            If no value is passed, the default `predict` interface is used.

        optimization: {"fidelity", "accuracy"}, default="fidelity"
            The comparison criteria to optimize the decision tree students in Trustee inner-loop.
            Used for ablation study only.

        aggregate: bool, default=True
            Boolean indicating whether dataset aggregation should be used in Trustee inner-loop.
            Used for ablation study only.

        verbose: bool, default=False
            Boolean indicating whether to log messages.
        """
        if verbose:
            self.log(f"Initializing training dataset using {self.expert} as expert model")

        if len(X) != len(y):
            raise ValueError("Features (X) and target (y) values should have the same length.")


        print(X)
        print(type(X))

        # Your feature names

        # # Convert
        # river_data = dataframe_to_river_dicts(X, feature_names)

        # # Loop through samples
        # for sample in river_data:
        #     prediction = getattr(self.expert, predict_method_name)(sample)      # make prediction
        #     print("Prediction:", prediction)

        # reconstructed = reconstruct_encoded_generator(X)
        # for predicted_sample in getattr(self.expert, predict_method_name)(reconstructed):
        #     print(predicted_sample)


        # convert data to np array to facilitate processing
        X = convert_to_df(X)
        y = convert_to_series(y)

        self._student_use_features = use_features if use_features is not None else np.arange(0, len(X.columns))

        # split input array to train DTs and evaluate agreement
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, train_size=train_size)

        features = self._X_train

        print("StreamTrustee.fit() - starting")
        print("Predict method:", predict_method_name)
        print("X_train type:", type(self._X_train))
        print("X_train[0]:", self._X_train.iloc[0] if len(self._X_train) > 0 else "empty")

        # # RMC: changed line below for one compatible with River
        # if predict_method_name == "predict_one":
        #     #raw_preds = [self.expert.predict_one(row.to_dict()) for _, row in self._X_train.iterrows()]
        #     raw_preds = []
        #     for i, row in self._X_train.iterrows():
        #         input_dict = row.to_dict()
        #         if i < 5:
        #             print(f"Input to predict_one at row {i}: {input_dict}")
        #         pred = self.expert.predict_one(input_dict)
        #         if i < 5:
        #             print(f"Prediction from predict_one at row {i}: {pred}")
        #         raw_preds.append(pred)

        # else:
        #     raw_preds = getattr(self.expert, predict_method_name)(self._X_train)

        # # # RMC Previous version of adaptation of Trustee to River
        # # if self.predict_method_name == "predict_one":
        # #     y_pred = pd.Series([blackbox_copy.predict_one(row.to_dict()) for _, row in X_test.iterrows()])
        # # else:
        # #     y_pred = getattr(blackbox_copy, self.predict_method_name)(X_test)


        # # # ===== INSERT DEBUG PRINTS HERE =====
        # # print("Debug: first 10 training rows and expert predictions:")
        # # for i in range(min(10, len(self._X_train))):
        # #     print(f"Row {i}: {self._X_train.iloc[i].to_dict()}")
        # #     print(f"Expert prediction: {raw_preds[i]}")

        # # Force binary labels: anything other than 1 becomes 0
        # targets = pd.Series([1 if pred == 1 else 0 for pred in raw_preds])



        if predict_method_name == "predict_one":
            print("@@@ Methods called in 2: main.py")
            print([m for m in dir(self.expert) if callable(getattr(self.expert, m))])
            print("Structure of X_train")
            print(self._X_train)
            print(type(self._X_train))
            print(self.expert)
            print(type(self.expert))
            # targets = pd.Series([self.expert.predict_one(row.to_dict()) for _, row in self._X_train.iterrows()])
            targets = pd.Series([
                self.expert.predict_one(dict(zip(feature_names, row)))
                for row in self._X_train.values
            ])
        else:
            # Debug / check shapes and types before calling explain_with_*
            print("@@@ Methods called in 2: main.py")
            print([m for m in dir(self.expert) if callable(getattr(self.expert, m))])
            targets = convert_to_series(getattr(self.expert, predict_method_name)(self._X_train))

        print("Predict done in main.")

        # 🔍 Sanity check on target values
        print("Sanity check: initial `targets` values and types")
        print(targets.value_counts(dropna=False))
        print("targets dtype:", targets.dtype)
        print("targets unique values:", targets.unique())
        
        # print("Unique labels in initial targets:", targets.unique())

        if hasattr(targets, "shape") and len(targets.shape) >= 2:
            targets = targets.ravel()

        student = self.student_class(
            random_state=0, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, ccp_alpha=ccp_alpha
        )

        if verbose:
            self.log(f"Expert model score: {self._score(self._y_train, targets)}")
            self.log(f"Initializing Trustee outer-loop with {num_stability_iter} iterations")

        # Trustee outer-loop
        for i in range(num_stability_iter):
            self._students_by_iter.append([])
            if verbose:
                self.log("#" * 10, f"Outer-loop Iteration {i}/{num_stability_iter}", "#" * 10)
                self.log(f"Initializing Trustee inner-loop with {num_stability_iter} iterations")

            # Trustee inner-loop
            for j in range(num_iter):
                if verbose:
                    self.log("#" * 10, f"Inner-loop Iteration {j}/{num_iter}", "#" * 10)

                dataset_size = len(features)
                size = int(int(len(self._X_train)) * samples_size) if samples_size else num_samples
                # Step 1: Sample predictions from training dataset
                if verbose:
                    self.log(
                        f"Sampling {size} points from training dataset with ({len(features)}, {len(targets)}) entries"
                    )

                samples_idxs = np.random.choice(dataset_size, size=size, replace=False)
                X_iter, y_iter = features.iloc[samples_idxs], targets.iloc[samples_idxs]
                X_iter_train, X_iter_test, y_iter_train, y_iter_test = train_test_split(
                    X_iter, y_iter, train_size=train_size
                )
                X_train_student = X_iter_train.iloc[:, self._student_use_features]
                X_test_student = X_iter_test.iloc[:, self._student_use_features]

                # Step 2: Training DecisionTreeRegressor with sampled data
                student.fit(X_train_student.values, y_iter_train.values)
                student_pred = student.predict(X_test_student.values)

                if verbose:
                    self.log(
                        f"Student model {i}-{j} trained with depth {student.get_depth()} "
                        f"and {student.get_n_leaves()} leaves:"
                    )
                    self.log(f"Student model score: {self._score(y_iter_test, student_pred)}")

                # Step 3: Use expert model predictions to aggregate original dataset
                if predict_method_name == "predict_one":
                    expert_pred_list = []
                    for k, (_, row) in enumerate(X_iter_test.iterrows()):
                        # Build dict using mandatory feature_names list
                        sample = dict(zip(feature_names, row.values))
                        pred = self.expert.predict_one(sample)
                        if pred not in [0, 1]:
                            print(f"[DEBUG] Forcing expert prediction {pred} → 0 at inner-loop index {k}")
                            pred = 0
                        expert_pred_list.append(pred)
                    expert_pred = pd.Series(expert_pred_list)
                else:
                    expert_pred = pd.Series(getattr(self.expert, predict_method_name)(X_iter_test))

                # print("Unique labels in expert_pred:", expert_pred.unique())

                if hasattr(expert_pred, "shape") and len(expert_pred.shape) >= 2:
                    expert_pred = expert_pred.ravel()

                if aggregate:
                    features = pd.concat([features, X_iter_test])
                    targets = pd.concat([targets, expert_pred])

                if optimization == "accuracy":
                    reward = self._score(y_iter_test, student_pred)
                else:
                    reward = self._score(expert_pred, student_pred)

                if verbose:
                    self.log(f"Student model {i}-{j} fidelity: {reward}")

                self._students_by_iter[i].append((deepcopy(student), reward))

            self._top_students.append(max(self._students_by_iter[i], key=lambda item: item[1]))

        self._best_student = self.explain(top_k=top_k)[0]



    @_check_if_trained
    def explain(self, top_k=10):
        """
        Returns explainable model that best imitates Expert model, based on highest mean agreement and highest fidelity.

        Returns
        -------
        top_student: tuple
            (dt, pruned_dt, agreement, reward)

            - dt: {DecisionTreeClassifier, DecisionTreeRegressor}
                Unconstrained fitted student model.

            - pruned_dt: {DecisionTreeClassifier, DecisionTreeRegressor}
                Top-k pruned fitted student model.

            - agreement: float
                Mean agreement of pruned student model with respect to others.

            - reward: float
                Fidelity of student model to the expert model.
        """
        stable = self.get_stable(top_k=top_k, threshold=0, sort=False)
        # calculate distance
        num_stables = len(stable)
        dt_distance = [[0] * num_stables for _ in range(num_stables)]
        feature_names = [f"{i}" for i in range(len(self._X_train[0]))]

        max_distance = 0
        for left in range(num_stables):
            for right in range(left, num_stables):
                left_tree_text = tree_representation.tree_to_bracket_notation(stable[left][0], feature_names,
                                                                              tree_representation.recurse_tree_with_classes)
                right_tree_text = tree_representation.tree_to_bracket_notation(stable[right][0], feature_names,
                                                                               tree_representation.recurse_tree_with_classes)

                left_tree, right_tree = map(Tree.from_text, [left_tree_text, right_tree_text])

                apted_instance = apted.APTED(left_tree, right_tree)
                ted = apted_instance.compute_edit_distance()
                dt_distance[left][right] = ted
                dt_distance[right][left] = ted
                max_distance = max(max_distance, ted)

        if max_distance == 0:
            return max(stable, key=lambda item: item[2])

        dt_distance_avg = [sum(dt_distance[i]) / len(dt_distance[i]) for i in range(num_stables)]
        dt_similarity_avg_norm = [1 - distance_avg / max_distance for distance_avg in dt_distance_avg]

        max_agreement_and_similarity = 0
        best_tree = None
        for idx, stable_tree in enumerate(stable):
            agreement_and_similarity = dt_similarity_avg_norm[idx] + stable_tree[2]
            if agreement_and_similarity > max_agreement_and_similarity:
                max_agreement_and_similarity = agreement_and_similarity
                best_tree = stable_tree

        return best_tree
        # return max(stable, key=lambda item: item[2])

    @_check_if_trained
    def get_stable(self, top_k=10, threshold=0.9, sort=True):
        """
        Filters out explanations from Trustee stability analysis with less than threshold agreement.

        Parameters
        ----------
        top_k: int, default=10
            Number of top-k branches, sorted by number of samples per branch, to keep after finding
            decision tree with highest fidelity.

        threshold: float, default=0.9
            Remove any student decision tree explanation if their mean agreement goes below given threshold.
            To keep all students regardless of mean agreement, pass 0.

        sort: bool, default=True
            Boolean indicating whether to sort returned stable student explanation based on mean agreement.

        Returns
        -------
        stable_explanations: array-like of tuple
            [(dt, pruned_dt, agreement, reward), ...]

            - dt: {DecisionTreeClassifier, DecisionTreeRegressor}
                Unconstrained fitted student model.

            - pruned_dt: {DecisionTreeClassifier, DecisionTreeRegressor}
                Top-k pruned fitted student model.

            - agreement: float
                Mean agreement of pruned student model with respect to others.

            - reward: float
                Fidelity of student model to the expert model.
        """
        if len(self._stable_students) == 0:
            agreement = []
            # Calculate pair-wise agreement of all top students generated during inner loop
            for i, _ in enumerate(self._top_students):
                agreement.append([])
                # Apply top-k pruning before calculating agreement
                base_tree = top_k_prune(self._top_students[i][0], top_k=top_k)
                for j, _ in enumerate(self._top_students):
                    # Apply top-k pruning before calculating agreement
                    iter_tree = top_k_prune(self._top_students[j][0], top_k=top_k)

                    iter_y_pred = iter_tree.predict(self._X_test.iloc[:, self._student_use_features].values)
                    base_y_pred = base_tree.predict(self._X_test.iloc[:, self._student_use_features].values)

                    agreement[i].append(self._score(iter_y_pred, base_y_pred))

                # Save complete dt, top-k prune dt, mean agreement and fidelity
                self._stable_students.append(
                    (
                        self._top_students[i][0],
                        base_tree,
                        np.mean(agreement[i]),
                        self._top_students[i][1],
                    )
                )

        stable = self._stable_students
        if threshold > 0:
            stable = filter(lambda item: item[2] >= threshold, stable)

        if sort:
            return sorted(stable, key=lambda item: item[2], reverse=True)

        return stable

    @_check_if_trained
    def get_all_students(self):
        """
        Get list of all (student, reward) obtained during the inner-loop process.

        Returns
        -------
        students_by_iter: array-like of shape (num_stability_iter, num_iter) of tuple (dt, reward)
            Matrix with all student models trained during `fit()`.

            - dt: {DecisionTreeClassifier, DecisionTreeRegressor}
                Unconstrained fitted student model.

            - reward: float
                Fidelity of student model to the expert model.
        """
        return self._students_by_iter

    @_check_if_trained
    def get_top_students(self):
        """
        Get list of top (students, reward) obtained during the outer-loop process.

        Returns
        -------
        top_students: array-like of shape (num_stability_iter,) of tuple (dt, reward)
            List with top student models trained during `fit()`.

            - dt: {DecisionTreeClassifier, DecisionTreeRegressor}
                Unconstrained fitted student model.

            - reward: float
                Fidelity of student model to the expert model.
        """
        return self._top_students

    @_check_if_trained
    def get_n_features(self):
        """
        Returns number of features used in the top student model.

        Returns
        -------
        n_features: int
            Number of features used in top student model.
        """
        if not self._features:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return len(self._features.keys())

    @_check_if_trained
    def get_n_classes(self):
        """
        Returns number of classes used in the top student model.

        Returns
        -------
        n_classes: int
            Number of classes outputted in top student model.
        """
        return self._best_student.tree_.n_classes[0]

    @_check_if_trained
    def get_samples_sum(self):
        """
        Get the sum of all samples in all non-leaf _nodes in best student model.

        Returns
        -------
        samples_sum: int
            Sum of all samples covered by non-leaf nodes in top student model.
        """
        left = self._best_student.tree_.children_left
        right = self._best_student.tree_.children_right
        samples = self._best_student.tree_.n_node_samples

        return np.sum([n_samples if left[node] != right[node] else 0 for node, n_samples in enumerate(samples)])

    @_check_if_trained
    def get_top_branches(self, top_k=10):
        """
        Returns list of top-k _branches of the best student, sorted by the number of samples the branch classifies.

        Parameters
        ----------
        top_k: int, default=10
            Number of top-k branches, sorted by number of samples per branch, to return.

        Returns
        -------
        top_branches: array-like of dict
            Dict of top-k branches from top student model.

            - dict: { "level": int, "path": array-like of dict, "class": int, "prob": float, "samples": int}
        """
        if not self._branches:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return sorted(self._branches, key=lambda p: p["samples"], reverse=True)[:top_k]

    @_check_if_trained
    def get_top_features(self, top_k=10):
        """
        Get list of top _features of the best student, sorted by the number of samples the feature is used to classify.

        Parameters
        ----------
        top_k: int, default=10
            Number of top-k features, sorted by number of samples per branch, to return.


        Returns
        -------
        top_features: array-like of dict
            List of top-k features from top student model.

            - dict {"<feature>(int)" : {"count": int"samples": int}}
        """
        if not self._features:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return sorted(self._features.items(), key=lambda p: p[1]["samples"], reverse=True)[:top_k]

    @_check_if_trained
    def get_top_nodes(self, top_k=10):
        """
        Returns list of top _nodes of the best student, sorted by the proportion of samples split by each node.

        The proportion of samples is calculated based on the impurity decrease equation is the following::
            n_samples * abs(left_impurity - right_impurity)

        Parameters
        ----------
        top_k: int, default=10
            Number of top-k nodes, sorted by number of samples per branch, to return.

        Returns
        -------
        top_nodes: array-like of dict
            List of top-k nodes from top student model.

            - dict: {"idx": int, "level": int, "feature": int, "threshold": float, "samples": int,
                     "values": tuple of int, "gini_split": tuple of float, "data_split": tuple of float,
                     "data_split_by_class": array-like of tuple of float}
        """
        if not self._nodes:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return sorted(
            self._nodes, key=lambda p: p["samples"] * abs(p["gini_split"][0] - p["gini_split"][1]), reverse=True
        )[:top_k]

    @_check_if_trained
    def get_samples_by_level(self):
        """
        Get number of samples by level of the best student.

        Returns
        -------
        samples_by_level: dict of int
            Dict of samples by level. {"<level>(int)": <samples>(int)}
        """
        if not self._nodes:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        samples_by_level = list(np.zeros(self._best_student.get_depth() + 1))
        for node in self._nodes:
            samples_by_level[node["level"]] += node["samples"]

        for node in self._branches:
            samples_by_level[node["level"]] += node["samples"]

        return samples_by_level

    @_check_if_trained
    def get_leaves_by_level(self):
        """
        Returns number of leaves by level of the best student.

        Returns
        -------
        leaves_by_level: dict of int
            Dict of leaves by level. {"<level>(int)": <leaves>(int)}
        """
        if not self._branches:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        leaves_by_level = list(np.zeros(self._best_student.get_depth() + 1).astype(int))
        for node in self._branches:
            leaves_by_level[node["level"]] += 1

        return leaves_by_level

    @_check_if_trained
    def prune(self, top_k=10, max_impurity=0.10):
        """
        Prunes and returns the best student model explanation from the list of _students_by_iter.

        Parameters
        ----------
        top_k: int, default=10
            Number of top-k branches, sorted by number of samples per branch, to return.

        max_impurity: float, default=0.10
            Maximum impurity allowed in a branch. Will prune anything below that impurity level.

        Returns
        -------
        top_k_pruned_student: {DecisionTreeClassifier, DecisionTreeRegressor}
            Top-k pruned best fitted student model.
        """
        return top_k_prune(self._best_student, top_k=top_k, max_impurity=max_impurity)


class ClassificationTrustee(Trustee):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train
    a student DecisionTreeClassifier based on observations from an Expert classification model.
    """

    def __init__(self, expert, logger=None):
        """
        Classification Trustee constructor

        Parameters
        ----------
        expert: object
            The ML blackbox model to analyze. The expert model must have a `predict` method call implemented for
            Trustee to work properly, unless explicitly stated otherwise using the `predict_method_name` argument
            in the fit() method.
        logger: Logger object , default=None
            A logger object to log messages to. If none is given, the print() method will be used to log messages.
        """
        super().__init__(expert, student_class=DecisionTreeClassifier, logger=logger)

    def _score(self, y_true, y_pred, average="macro"):
        """
        Score function for student models. Compares the ground-truth predictions
        of a blackbox model with the predictions of a student model, using F1-score.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The ground-truth target values (class labels in classification, real numbers in regression).

        y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values (class labels in classification, real numbers in regression).

        Returns
        -------
        score: float
            Calculated F1-score between student model predictions and expert model ground-truth.
        """
        return f1_score(y_true, y_pred, average=average)


class RegressionTrustee(Trustee):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train a
    student DecisionTreeRegressor based on observations from an Expert regression model.
    """

    def __init__(self, expert, logger=None):
        """
        Regression Trustee constructor

        Parameters
        ----------
        expert: object
            The ML blackbox model to analyze. The expert model must have a `predict` method call implemented for
            Trustee to work properly, unless explicitly stated otherwise using the `predict_method_name` argument
            in the fit() method.
        logger: Logger object , default=None
            A logger object to log messages to. If none is given, the print() method will be used to log messages.
        """
        super().__init__(expert=expert, student_class=DecisionTreeRegressor, logger=logger)

    def _score(self, y_true, y_pred):
        """
        Score function for student models. Compares the ground-truth predictions
        of a blackbox model with the predictions of a student model, using R2-score.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The ground-truth target values (class labels in classification, real numbers in regression).

        y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values (class labels in classification, real numbers in regression).

        Returns
        -------
        score: float
            Calculated R2-score between student model predictions and expert model ground-truth.
        """
        return r2_score(y_true, y_pred)
