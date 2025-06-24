from abc import ABC, abstractmethod


class IShapExplainableModel(ABC):
    """
    Generic interface for SHAP explainable models to implement.
    """

    @abstractmethod
    def explain_with_shap(self, **kwargs):
        """
        Initializes the explainer of the algorithms
        """
        pass
