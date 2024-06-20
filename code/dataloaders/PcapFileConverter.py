import subprocess
import time
from typing import List, Tuple, Generator, Dict, Any

import common.global_variables as global_variables
from common.functions import report_performance
from dataloaders.IDataLoader import IDataLoader
from common.features import IFeature, PacketFeature
from itertools import islice  # Importing islice here
import random



from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


class PcapFileConverter(IDataLoader):
    def __init__(self, filepath: str, packet_processor_path: str, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.preprocessor_path = packet_processor_path
        log.info(f"[{ type(self).__name__ }] Reading from file: {self.filepath}")

    def get_samples(
        self,
    ) -> Generator[Dict[IFeature, Any], None, None]:
        pcap_call = [self.preprocessor_path, "stream-file", self.filepath]

        log.info(f"[PcapFileLoader] Processing file: {self.filepath}")
        sum_processing_time = 0
        packet_count = 0
        process = subprocess.Popen(
            pcap_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )

        while True:
            start_time_ref = time.process_time_ns()
            if process.poll() and process.returncode:
                log.error(process.stdout.readlines())
                raise RuntimeError(f"PCAP feature extractor exited with error code {process.returncode}!")
            packet_features = {
                PacketFeature.CPP_FEATURE_STRING: process.stdout.readline()
            }
            sum_processing_time += time.process_time_ns() - start_time_ref
            if packet_features[PacketFeature.CPP_FEATURE_STRING]:
                yield packet_features
                packet_count += 1
            else:
                break

        report_performance(type(self).__name__, log, packet_count, sum_processing_time)

        # Data loaders only exists once per data source, therefore they are
        # suitable for tracking the overall number of packets processed. This
        # value will be reported by the main pipeline in the end.
        global_variables.global_pipeline_packet_count += packet_count


    def randomize_and_sample_packets(self, new_feature_streams: List[Generator[Dict[Any, Any], None, None]], n_samples_per_subdataset: List[int]) -> Tuple[Generator[Dict[Any, Any], None, None], List[int]]:
        """
        Randomly samples a specified number of packets from each subdataset and returns them in a random order.

        Parameters:
        - new_feature_streams: List of generators, each producing packet samples from a subdataset.
        - n_samples_per_subdataset: List of integers specifying how many samples to take from each subdataset.

        Returns:
        - A tuple containing:
        - A generator producing the packet samples in a random order.
        - A list of integers representing the order of subdataset indices from which the samples were taken.
        """

        collected_samples = []
        index_vector = []

        # Step 1: Collect samples from each subdataset and maintain the index vector
        for i, (stream, n_samples) in enumerate(zip(new_feature_streams, n_samples_per_subdataset)):
            sampled_data = list(islice(stream, n_samples))  # Collect up to n_samples packets
            collected_samples.extend(sampled_data)  # Extend the collected_samples list
            index_vector.extend([i] * len(sampled_data))  # Extend the index_vector list with the dataset index

        # Step 2: Pair the samples with their indices and shuffle
        paired_samples = list(zip(collected_samples, index_vector))
        random.shuffle(paired_samples)

        # Step 3: Unzip the shuffled pairs into separate lists
        shuffled_samples, shuffled_indices = zip(*paired_samples) if paired_samples else ([], [])

        # Step 4: Return a generator for the samples and the list of indices
        return (sample for sample in shuffled_samples), list(shuffled_indices)


    @staticmethod
    def feature_signature() -> List[IFeature]:
        return [PacketFeature.CPP_FEATURE_STRING]
