from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score, classification_report
from river import metrics
import pandas as pd

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from datetime import datetime


class StreamClassifReporter(IReporter):
    """
    Can only be used when PredictionField.GROUND_TRUTH is known!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ground_truths = []
        self.predicted_labels = []
        self.model_name = kwargs.get('model_name', 'noname')

    def set_model_name(self, name: str):
        self.model_name = name

    def report(self, features: Dict[IFeature, Any]):
        self.ground_truths.append(features[PredictionField.GROUND_TRUTH])
        self.predicted_labels.append(features[PredictionField.OUTPUT_BINARY])

    def report_eval(self, y: int, y_pred: int):
        self.ground_truths.append(y)
        self.predicted_labels.append(y_pred)


    def _plot_pred_time_series(self):

        # Example data
        timestamps = np.arange(0, len(self.predicted_labels))  # Example: time points from 0 to 99
        pred_labels = np.array(self.predicted_labels)
        ground_truths = np.array(self.ground_truths)

        # Plotting the time-series scores
        plt.figure(figsize=(10, 6))  # Create a figure with a specific size

        plt.plot(timestamps, pred_labels, label='Predictions', color='b', linestyle='-', marker='.', markersize=0.5)
        plt.plot(timestamps, ground_truths, label='Ground truths', color='r', linestyle='-', marker='.', markersize=0.5)
        # plt.ylim(-10, 110)

        # Adding labels and title
        plt.xlabel('Sample')
        plt.ylabel('Prediction')
        plt.title('Time-Series Predictions')

        # Adding a grid for better readability
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(25))
        plt.grid(True)
        plt.tight_layout()

        # Optional: Highlighting thresholds or specific anomalies
        # Example: Highlight scores above a threshold (e.g., 0.8)
        # threshold = 0.230
        # high_anomalies = self.predicted_labels > threshold
        # plt.plot(timestamps[high_anomalies], self.predicted_labels[high_anomalies], 'ro', label='High Anomalies')
        # plt.plot(y=anomaly_threshold, color='r', linestyle='--', linewidth=1, label='Threshold')

        # Add a legend
        plt.legend()

        # Save the plot to a file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'configurations/zplots'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{current_time}_{self.model_name}_pred_timeseries.png"))
        plt.close()


    def _plot_cumulative_accuracy(self):
        """
        Computes and plots cumulative accuracy over time based on internal predictions and ground truths.
        """
        

        if not self.ground_truths or not self.predicted_labels:
            print("No data to plot cumulative accuracy.")
            return

        acc_metric = metrics.Accuracy()
        cumulative_accuracies = []

        for y_true, y_pred in zip(self.ground_truths, self.predicted_labels):
            acc_metric.update(y_true, y_pred)
            cumulative_accuracies.append(acc_metric.get() * 100)

        # Plot cumulative accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_accuracies, color='blue', linewidth=2)
        plt.ylim(-10, 110)
        plt.xlabel('Data Points Processed', fontsize=24)
        plt.ylabel('Cumulative Accuracy (%)', fontsize=24)
        plt.title('Cumulative Accuracy Over Time')
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(25))
        plt.grid(True)
        plt.tight_layout()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'configurations/zplots'
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(os.path.join(output_dir, f"{current_time}_{self.model_name}_cumulative_accuracy.png"))
        plt.close()
            
    def _plot_cumulative_accuracy_window(self, n_samples: int = 5):
        """
        Computes and plots sliding window accuracy over time based on internal predictions and ground truths.

        Parameters:
            n_samples (int): Size of the sliding window to compute accuracy.
        """

        if not self.ground_truths or not self.predicted_labels:
            print("No data to plot windowed cumulative accuracy.")
            return

        cumulative_accuracies = []

        for i in range(len(self.ground_truths)):
            start = max(0, i - n_samples + 1)
            gt_window = self.ground_truths[start:i+1]
            pred_window = self.predicted_labels[start:i+1]

            correct = sum(1 for gt, pred in zip(gt_window, pred_window) if gt == pred)
            acc = (correct / len(gt_window)) * 100
            cumulative_accuracies.append(acc)

        # Plot sliding window accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_accuracies, color='green', linewidth=2)
        plt.ylim(-10, 110)
        plt.xlabel('Data Points Processed', fontsize=24)
        plt.ylabel('Window Accuracy (%)', fontsize=24)
        plt.title(f'Cumulative Accuracy (Window={n_samples})')
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(25))
        plt.grid(True)
        plt.tight_layout()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'configurations/zplots'
        os.makedirs(output_dir, exist_ok=True)

        plot_path = os.path.join(output_dir, f"{current_time}_{self.model_name}_cumulative_window_{n_samples}.png")
        plt.savefig(plot_path)
        plt.close()

    def _sanitize_predictions(self):
        self.predicted_labels = [0 if v is None else v for v in self.predicted_labels]

    def _get_unique_labels(self):
        return sorted(set(self.ground_truths + self.predicted_labels))

    def _validate_or_default_label_map(self, label_map, labels):
        if label_map is None or len(label_map) != len(labels):
            return [str(lbl) for lbl in labels]
        return label_map

    def _get_label_name(self, label, labels, label_map):
        try:
            idx = labels.index(label)
            return label_map[idx]
        except ValueError:
            return str(label)

    def _compute_metrics(self, labels):
        cnf_matrix = confusion_matrix(self.ground_truths, self.predicted_labels, labels=labels)
        accuracy = accuracy_score(self.ground_truths, self.predicted_labels)
        precision = precision_score(self.ground_truths, self.predicted_labels, average='macro', zero_division=0)
        recall = recall_score(self.ground_truths, self.predicted_labels, average='macro', zero_division=0)
        f1 = f1_score(self.ground_truths, self.predicted_labels, average='macro', zero_division=0)
        return cnf_matrix, (accuracy, precision, recall, f1)

    def _write_label_predictions_to_file(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f'configurations/zplots/{current_time}_{self.model_name}_features_scores_labels.txt'
        os.makedirs('configurations/zplots', exist_ok=True)
        with open(path, "w") as f:
            f.write("Label, Prediction\n")
            for gt, pred in zip(self.ground_truths, self.predicted_labels):
                f.write(f"{gt}, {pred}\n")

    def _plot_confusion_matrix(self, cnf_matrix, labels, label_map, metrics_scores):
        accuracy, precision, recall, f1 = metrics_scores
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        sns.set(font_scale=2.5)
        sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted Labels', fontsize=20)
        ax.set_ylabel('True Labels', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=1, bottom=0.35, right=1, left=0.07)
        metrics_text = f"Accuracy: {accuracy*100:.2f}%\nPrecision: {precision*100:.2f}%\nRecall: {recall*100:.2f}%\nF1 Score: {f1*100:.2f}%"
        label_legend = '\n'.join([f"{lbl}: {name}" for lbl, name in zip(labels, label_map)])
        plt.figtext(0.1, 0.05, metrics_text, ha='left', fontsize=28)
        plt.figtext(0.7, 0.05, label_legend, ha='left', fontsize=25)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f'configurations/zplots/{current_time}_{self.model_name}_confusion_matrix.png'
        plt.savefig(path)
        plt.close()

    def _write_global_report(self, label_map):
        report_dict = classification_report(self.ground_truths, self.predicted_labels, output_dict=True, target_names=label_map)
        report_str = classification_report(self.ground_truths, self.predicted_labels, digits=4, target_names=label_map)
        df_global = pd.DataFrame(report_dict).transpose()
        return report_str, df_global

    def _compute_phase_reports(self, n_subsets_per_concept, samples_per_subset, labels, label_map, get_label_name):
        reports = []
        index = 0
        subset_idx = 0
        for concept_idx, n_subsets in enumerate(n_subsets_per_concept):
            total_samples = sum(samples_per_subset[subset_idx:subset_idx + n_subsets])
            gt_window = self.ground_truths[index:index + total_samples]
            pred_window = self.predicted_labels[index:index + total_samples]
            phase_label = next(l for l in set(gt_window) if l != 0)
            bin_gt = [1 if x == phase_label else 0 for x in gt_window]
            bin_pred = [1 if x == phase_label else 0 for x in pred_window]
            report_dict = classification_report(bin_gt, bin_pred, output_dict=True)
            report_str = classification_report(bin_gt, bin_pred, digits=4)
            df = pd.DataFrame(report_dict).transpose()
            acc = accuracy_score(bin_gt, bin_pred)
            reports.append({
                "phase": concept_idx + 1,
                "label": phase_label,
                "label_name": get_label_name(phase_label),
                "accuracy": acc,
                "report_dict": report_dict,
                "report_str": report_str,
                "report_df": df
            })
            index += total_samples
            subset_idx += n_subsets
        return reports

    def _write_phase_report(self, report_str, phase_reports, df_global, get_label_name):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f'configurations/zplots/{current_time}_{self.model_name}_multiclass_confusion_matrix.txt'
        with open(path, "w") as f:
            f.write("Global Per-class Classification Metrics\n")
            f.write("========================================\n\n")
            f.write(report_str + "\n\n")
            f.write("Phase-wise Binary Classification Metrics\n")
            f.write("========================================\n\n")
            for report in phase_reports:
                f.write(f"Phase {report['phase']} (Label {report['label']} - {report['label_name']} vs Benign)\n")
                f.write("----------------------------------------\n")
                df = report["report_df"]
                if "1" in df.index:
                    row = df.loc["1"]
                    f.write(f"Precision: {row['precision']:.4f}\nRecall:    {row['recall']:.4f}\n"
                            f"F1-score:  {row['f1-score']:.4f}\nSupport:   {int(row['support'])}\n"
                            f"Accuracy:  {report['accuracy']:.4f}\n\n")
                else:
                    f.write("Warning: No positive class ('1') found.\n\n")

    def _plot_metrics_tables(self, phase_reports, df_global, get_label_name):
        df_global_fmt = df_global.drop("accuracy", errors="ignore").copy()
        df_global_fmt["Support"] = df_global_fmt["support"].astype(int)
        df_global_fmt = df_global_fmt[["precision", "recall", "f1-score", "Support"]].rename(
            columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score"}
        )
        summary_rows = []
        for report in phase_reports:
            df = report["report_df"]
            if "1" in df.index:
                row = df.loc["1"]
                summary_rows.append({
                    "Phase": report["phase"],
                    "Label": report["label"],
                    "Label Name": report["label_name"],
                    "Accuracy": report["accuracy"],
                    "Precision": row["precision"],
                    "Recall": row["recall"],
                    "F1-score": row["f1-score"],
                    "Support": int(row["support"]),
                })
        df_phase_summary = pd.DataFrame(summary_rows)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle("Classification Metrics Summary", fontsize=16)
        axes[0].axis('off')
        global_table = axes[0].table(
            cellText=np.round(df_global_fmt.values, 4),
            rowLabels=[get_label_name(label) for label in df_global_fmt.index],
            colLabels=df_global_fmt.columns,
            loc='center'
        )
        global_table.auto_set_font_size(False)
        global_table.set_fontsize(10)
        axes[0].set_title("Global Per-Class Metrics", fontsize=14)
        axes[1].axis('off')
        phase_table = axes[1].table(
            cellText=np.round(df_phase_summary.drop(columns="Label Name").values, 4),
            colLabels=[col for col in df_phase_summary.columns if col != "Label Name"],
            loc='center'
        )
        phase_table.auto_set_font_size(False)
        phase_table.set_fontsize(10)
        axes[1].set_title("Per-Phase Binary Metrics", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'configurations/zplots/{current_time}_{self.model_name}_metrics_tables.png', dpi=300)
        plt.close()

    def _log_summary(self, cnf_matrix, labels, metrics_scores):
        accuracy, precision, recall, f1 = metrics_scores
        log = PipelineLogger.get_logger()
        log.info(f"\n---\nReport\n\nConfusion matrix:\n{cnf_matrix}\n\n"
                f"Labels: {labels}\n"
                f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n---")

    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
        ]

    def end_processing(self, n_subsets_per_concept: List[int], samples_per_subset: List[int], label_map: List[str] = None):
        self._sanitize_predictions()
        labels = self._get_unique_labels()
        label_map = self._validate_or_default_label_map(label_map, labels)
        get_label_name = lambda lbl: self._get_label_name(lbl, labels, label_map)

        cnf_matrix, metrics_scores = self._compute_metrics(labels)
        self._write_label_predictions_to_file()
        self._plot_confusion_matrix(cnf_matrix, labels, label_map, metrics_scores)
        report_str, df_global = self._write_global_report(label_map)

        phase_reports = self._compute_phase_reports(n_subsets_per_concept, samples_per_subset, labels, label_map, get_label_name)
        self._write_phase_report(report_str, phase_reports, df_global, get_label_name)
        self._plot_metrics_tables(phase_reports, df_global, get_label_name)
        self._plot_pred_time_series()
        self._plot_cumulative_accuracy()
        self._plot_cumulative_accuracy_window()
        self._log_summary(cnf_matrix, labels, metrics_scores)
