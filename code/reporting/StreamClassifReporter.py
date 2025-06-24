from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle


class StreamClassifReporter(IReporter):
    """
    Can only be used when PredictionField.GROUND_TRUTH is known!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ground_truths = []
        self.predicted_labels = []
        self.model_name = kwargs.get('model_name', 'noname')

    def set_model_name(self, name:str):
        self.model_name = name

    def report(self, features: Dict[IFeature, Any]):
        self.ground_truths.append(features[PredictionField.GROUND_TRUTH])
        self.predicted_labels.append(features[PredictionField.OUTPUT_BINARY])

    def report_eval(self, y: int, y_pred: int):
        self.ground_truths.append(y)
        self.predicted_labels.append(y_pred)

    def end_processing(self, n_subsets_per_concept: List[int], samples_per_subset: List[int], label_map=None):

        log = PipelineLogger.get_logger()

        # Check for None in self.predicted_labels and replace Nones for 0s
        none_in_predicted_labels = [i for i, value in enumerate(self.predicted_labels) if value is None]
        for i in none_in_predicted_labels:
            self.predicted_labels[i] = 0

        labels = sorted(set(self.ground_truths + self.predicted_labels))

        labelsName = [str(lbl) for lbl in labels]
        # if label_map is None:
        #     labelsName = [str(lbl) for lbl in labels]
        # else:
        #     labelsName = [label_map[k] for k in labels]

        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(self.ground_truths, self.predicted_labels, labels=labels)


        ### Calculating performance metrics
        # # Binary
        # TN, FP, FN, TP = cnf_matrix.ravel()
        # accuracy = (TP+TN)/(TP+TN+FP+FN)
        # precision = (TP)/(TP+FP)
        # recall = (TP)/(TP+FN)
        # f1 = 2*(precision*recall)/(precision+recall)
        # # Multiclass
        accuracy = accuracy_score(self.ground_truths, self.predicted_labels)
        precision = precision_score(self.ground_truths, self.predicted_labels, average='macro', zero_division=0)
        recall = recall_score(self.ground_truths, self.predicted_labels, average='macro', zero_division=0)
        f1 = f1_score(self.ground_truths, self.predicted_labels, average='macro', zero_division=0)

        ###########################################################################
        # Creating prefix of output files

        # Generate the file name with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = 'configurations/zplots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # imagepkl_path = os.path.join(output_dir, f'{current_time}_{self.model_name}_confusion_matrix.pkl')
        conf_image_path = os.path.join(output_dir, f'{current_time}_{self.model_name}_confusion_matrix.png')
        conf_text_path = os.path.join(output_dir, f'{current_time}_{self.model_name}_multiclass_confusion_matrix.txt')
        imagetime_path = os.path.join(output_dir, f'{current_time}_{self.model_name}_timeseriesanomaly.png')
        image_roc_auc_path = os.path.join(output_dir, f'{current_time}_{self.model_name}_roc_auc.png')
        text_scores_path = os.path.join(output_dir, f'{current_time}_{self.model_name}_features_scores_labels.txt')
        

        ###########################################################################
        ### Storing vectors of true values and predictions as columns in file

        with open(text_scores_path, "w") as file:
            
            # Print headers for readability (optional)
            file.write("Label, Prediction\n")
            
            # Iterate over the range of the maximum length
            for i in range(len(self.predicted_labels)):
                # Get elements or default to empty if the vector is shorter
                elem1 = self.ground_truths[i]
                elem2 = self.predicted_labels[i]
                
                # Write elements side by side with a tab separator
                file.write(f"{elem1}, {elem2}\n")

        ###########################################################################
        ### Printing confusion matrix in figure

        # Bigger figure width, moderate height (adjust as needed)
        fig = plt.figure(figsize=(12, 10))

        # Single axes for heatmap filling most of the figure
        ax = fig.add_subplot(111)

        sns.set(font_scale=2.5)
        sns.heatmap(
            cnf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labelsName,
            yticklabels=labelsName,
            ax=ax
        )

        ax.set_xlabel('Predicted Labels', fontsize=20)
        ax.set_ylabel('True Labels', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        # Remove default top and bottom margins as much as possible:
        plt.tight_layout()
        plt.subplots_adjust(top=1, bottom=0.35, right=1, left=0.07)  # Adjust bottom for text, top to reduce margin

        # Prepare the metrics text and label legend string
        metrics_text = (
            f"Accuracy: {accuracy * 100:.2f}%\n"
            f"Precision: {precision * 100:.2f}%\n"
            f"Recall:      {recall * 100:.2f}%\n"
            f"F1 Score: {f1 * 100:.2f}%"
        )

        label_legend = '\n'.join([f"{k}: {v}" for k, v in label_map.items()])

        # Add metrics and labels text *below* the heatmap using figtext with manual positioning
        plt.figtext(0.1, 0.05, metrics_text, ha='left', fontsize=28)
        plt.figtext(0.7, 0.05, label_legend, ha='left', fontsize=25)

        plt.savefig(conf_image_path)
        plt.close()


        
        ###################################################################################
        ### Printing confusion matrix in text file

        # Compute global classification report with target names
        report_dict = classification_report(
            self.ground_truths, 
            self.predicted_labels, 
            output_dict=True,
            target_names=[label_map[i] for i in sorted(label_map.keys())]
        )
        report_str = classification_report(
            self.ground_truths, 
            self.predicted_labels, 
            digits=4,
            target_names=[label_map[i] for i in sorted(label_map.keys())]
        )

        df_global = pd.DataFrame(report_dict).transpose()

        # Compute overall global accuracy
        global_accuracy = accuracy_score(self.ground_truths, self.predicted_labels)

        # Compute phase-wise accuracy (for all samples combined)

        # Store per-phase metrics in a list
        phase_reports = []

        index = 0
        subset_idx = 0

        for concept_idx, n_subsets in enumerate(n_subsets_per_concept):
            total_samples = sum(samples_per_subset[subset_idx:subset_idx + n_subsets])
            gt_window = self.ground_truths[index:index + total_samples]
            pred_window = self.predicted_labels[index:index + total_samples]

            # Determine the unique non-zero label in this phase
            unique_labels = set(gt_window)
            phase_label = next(l for l in unique_labels if l != 0)

            # Binarize: non-zero label = 1, benign (0) = 0
            bin_gt = [1 if x == phase_label else 0 for x in gt_window]
            bin_pred = [1 if x == phase_label else 0 for x in pred_window]

            # Compute classification report and accuracy for binary classification
            report_dict_phase = classification_report(bin_gt, bin_pred, output_dict=True)
            report_str_phase = classification_report(bin_gt, bin_pred, digits=4)
            df_phase = pd.DataFrame(report_dict_phase).transpose()
            phase_accuracy = accuracy_score(bin_gt, bin_pred)

            # Store phase report with label name
            phase_reports.append({
                "phase": concept_idx + 1,
                "label": phase_label,
                "label_name": label_map.get(phase_label, str(phase_label)),
                "accuracy": phase_accuracy,
                "report_dict": report_dict_phase,
                "report_str": report_str_phase,
                "report_df": df_phase
            })

            index += total_samples
            subset_idx += n_subsets


        # ---- PRINT TO FILE ----
        with open(conf_text_path, "w") as f:
            # ---- Global classification report ----
            f.write("Global Per-class Classification Metrics\n")
            f.write("========================================\n\n")
            f.write(report_str)
            f.write("\n\n")

            # ---- Per-phase binary reports ----
            f.write("Phase-wise Binary Classification Metrics\n")
            f.write("========================================\n\n")

            for report in phase_reports:
                f.write(f"Phase {report['phase']} (Label {report['label']} - {report['label_name']} vs Benign)\n")
                f.write("----------------------------------------\n")

                if "1" in report["report_df"].index:
                    row = report["report_df"].loc["1"]
                    f.write(f"Precision: {row['precision']:.4f}\n")
                    f.write(f"Recall:    {row['recall']:.4f}\n")
                    f.write(f"F1-score:  {row['f1-score']:.4f}\n")
                    f.write(f"Support:   {int(row['support'])}\n")
                    f.write(f"Accuracy:  {report['accuracy']:.4f}\n\n")
                else:
                    f.write("Warning: No positive class ('1') found in this phase's predictions.\n\n")

            # ---- Phase Table: Summary of metrics per phase ----
            f.write("Phase Summary Table (Text)\n")
            f.write("==========================\n\n")
            f.write(f"{'Phase':>5} {'Label':>6} {'Label Name':>12} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>6} {'Support':>8}\n")

            for report in phase_reports:
                df = report["report_df"]
                if "1" in df.index:
                    row = df.loc["1"]
                    f.write(f"{report['phase']:>5} {report['label']:>6} {report['label_name']:>12} {report['accuracy']:9.4f} "
                            f"{row['precision']:10.4f} {row['recall']:7.4f} {row['f1-score']:6.4f} {int(row['support']):8d}\n")
                else:
                    f.write(f"{report['phase']:>5} {report['label']:>6} {report['label_name']:>12} {'N/A':>9} {'N/A':>10} {'N/A':>7} {'N/A':>6} {'N/A':>8}\n")

            # ---- Global Classification Metrics (LaTeX) ----
            f.write("\n\n")
            f.write("Global Per-class Classification Metrics (LaTeX)\n")
            f.write("===============================================\n\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Class & Precision & Recall & F1-score & Support \\\\\n")
            f.write("\\midrule\n")

            # Map label indices to names for LaTeX output
            for label, row in df_global.iterrows():
                if label != 'accuracy':  # Skip 'accuracy' row (it's scalar)
                    class_name = label_map.get(label, str(label))
                    f.write(f"{class_name} & {row['precision']:.4f} & {row['recall']:.4f} "
                            f"& {row['f1-score']:.4f} & {int(row['support'])} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

            # ---- Phase Summary Table (LaTeX) ----
            f.write("\n\n")
            f.write("Phase Summary Table (LaTeX)\n")
            f.write("===========================\n\n")
            f.write("\\begin{tabular}{ccccccc}\n")
            f.write("\\toprule\n")
            f.write("Phase & Label & Label Name & Accuracy & Precision & Recall & F1-score & Support \\\\\n")
            f.write("\\midrule\n")

            for report in phase_reports:
                df = report["report_df"]
                if "1" in df.index:
                    row = df.loc["1"]
                    f.write(f"{report['phase']} & {report['label']} & {report['label_name']} & {report['accuracy']:.4f} & "
                            f"{row['precision']:.4f} & {row['recall']:.4f} & "
                            f"{row['f1-score']:.4f} & {int(row['support'])} \\\\\n")
                else:
                    f.write(f"{report['phase']} & {report['label']} & {report['label_name']} & N/A & N/A & N/A & N/A & N/A \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")


        # Extract phase summary as a DataFrame
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

        # Format global metrics table (df_global is from classification_report)
        df_global_fmt = df_global.drop("accuracy", errors="ignore").copy()
        df_global_fmt["Support"] = df_global_fmt["support"].astype(int)
        df_global_fmt = df_global_fmt[["precision", "recall", "f1-score", "Support"]].rename(
            columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score"}
        )

        # Plot side-by-side tables as an image
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle("Classification Metrics Summary", fontsize=16)

        # Global Table
        axes[0].axis('off')
        global_table = axes[0].table(cellText=np.round(df_global_fmt.values, 4),
                                    rowLabels=[label_map.get(label, str(label)) for label in df_global_fmt.index],
                                    colLabels=df_global_fmt.columns,
                                    loc='center')
        global_table.auto_set_font_size(False)
        global_table.set_fontsize(10)
        axes[0].set_title("Global Per-Class Metrics", fontsize=14)

        # Phase Summary Table
        axes[1].axis('off')
        phase_table = axes[1].table(cellText=np.round(df_phase_summary.drop(columns="Label Name").values, 4),
                                    colLabels=[col for col in df_phase_summary.columns if col != "Label Name"],
                                    loc='center')
        phase_table.auto_set_font_size(False)
        phase_table.set_fontsize(10)
        axes[1].set_title("Per-Phase Binary Metrics", fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        image_path = conf_text_path.replace(".txt", "_metrics_tables.png")
        plt.savefig(image_path, dpi=300)
        plt.close()



        ###########################################################################

        ##### PRINTING CONFUSION MATRIX RESULTS IN STDOUT

        log.info(f"\n---\nReport\n"
                 f"\nConfusion matrix:\n\n{cnf_matrix}\n\n"
                 f"Labels: {labels}\n"
                 f"(i-th row, j-th column: samples with true label i and predicted label j)\n\n"
                 f"Accuracy:"
                 f"{accuracy}\n"
                 f"Precision:"
                 f"{precision}\n"
                 f"Recall:"
                 f"{recall}\n"
                 f"F1 score: "
                 f"{f1}\n---"
                 )


    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
        ]
