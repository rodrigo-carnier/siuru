from abc import ABC, abstractmethod

class IExplainableAIAlgorithm(ABC):
    """
    Generic interface for XAI classes to implement.
    """
    @abstractmethod
    def explainer(self, **kwargs):
        """
        Initializes the explainer of the algorithms
        """
        pass



