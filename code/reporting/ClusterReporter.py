from collections import defaultdict
from typing import Dict, Any, List

from common.features import IFeature, PredictionField
from common.pipeline_logger import PipelineLogger
from reporting.IReporter import IReporter


class ClusterReporter(IReporter):
    """
    Can only be used when PredictionField.GROUND_TRUTH is known!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance_sum_per_model_and_label = defaultdict(lambda: 0)
        self.samples_per_model_and_label = defaultdict(lambda: 0)

    def report(self, features: Dict[IFeature, Any]):

        log.info("RMC: printing CReport.features(input)")
        print(features)

        # # Put the result into a color plot
        # Z = Z.reshape(xx.shape)
        # plt.figure(1)
        # plt.clf()
        # plt.imshow(
        #     Z,
        #     interpolation="nearest",
        #     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        #     cmap=plt.cm.Paired,
        #     aspect="auto",
        #     origin="lower",
        # )

        # plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # # Plot the centroids as a white X
        # centroids = kmeans.cluster_centers_
        # plt.scatter(
        #     centroids[:, 0],
        #     centroids[:, 1],
        #     marker="x",
        #     s=169,
        #     linewidths=3,
        #     color="w",
        #     zorder=10,
        # )
        # plt.title(
        #     "K-means clustering on the digits dataset (PCA-reduced data)\n"
        #     "Centroids are marked with white cross"
        # )
        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()




        # key = (
        #     features[PredictionField.MODEL_NAME],
        #     features[PredictionField.GROUND_TRUTH],
        # )
        # self.distance_sum_per_model_and_label[key] += features[
        #     PredictionField.OUTPUT_DISTANCE
        # ]
        # self.samples_per_model_and_label[key] += 1

    def end_processing(self):
        log = PipelineLogger.get_logger()
        report = "\n---\nDistance report\n"
        # for key, distance_sum in self.distance_sum_per_model_and_label.items():
        #     model, label = key
        #     avg = (
        #         self.distance_sum_per_model_and_label[key]
        #         / self.samples_per_model_and_label[key]
        #     )
        #     report += (
        #         f"Model: {model}\n"
        #         f"Label: {label}\n"
        #         f"Total samples: {self.samples_per_model_and_label[key]}\n"
        #         f"Average distance: {avg:.5E}\n\n"
        #     )
        log.info(report)

    @staticmethod
    def input_signature() -> List[IFeature]:
        return [
            PredictionField.MODEL_NAME,
            PredictionField.OUTPUT_BINARY,
            PredictionField.GROUND_TRUTH,
        ]
