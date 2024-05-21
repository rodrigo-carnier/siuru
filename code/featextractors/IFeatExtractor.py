from abc import ABC, abstractmethod
from typing import List

from common.features import IFeature, SampleGenerator


class IFeatExtractor(ABC):
    """
    Generic interface for feature extractor classes to implement.
    """

    @staticmethod
    @abstractmethod
    def input_signature() -> List[IFeature]:
        """
        Returns a list of features that the extractor requires in each input sample
        for internal processing.
        """
        pass

    @staticmethod
    @abstractmethod
    def output_signature() -> List[IFeature]:
        """
        Returns a list of features that the extractor promises to deliver
        (in addition to the existing features) in each sample when the generator
        is called.
        """
        pass

    @abstractmethod
    def extract(self, samples: SampleGenerator) -> SampleGenerator:
        """
        Applies feature extracting steps to samples in the input generator, then yields the
        modified samples. The number of yielded samples can be different from the input
        sample count!
        """
        pass
