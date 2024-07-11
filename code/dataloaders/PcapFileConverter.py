import subprocess
import time
from typing import List, Tuple, Generator, Dict, Any

import common.global_variables as global_variables
from common.functions import report_performance
from dataloaders.IDataLoader import IDataLoader
from common.features import IFeature, PacketFeature
from itertools import islice, chain
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
        self, **kwargs
    ) -> Generator[Dict[IFeature, Any], None, None]:
    # ) -> Tuple[Generator[Dict[Any, Any], None, None], List[int]]
        
        # TO DO: find a way to return packet_count in case we dont provide n_packet in **kwargs

        pcap_call = [self.preprocessor_path, "stream-file", self.filepath]

        log.info(f"[{ type(self).__name__ }] Processing file: {self.filepath}")
        sum_processing_time = 0
        packet_count = 0
        process = subprocess.Popen(
            pcap_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            # pcap_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )

        if kwargs.get('use_full_dataset', True):
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
        else:
            packets_to_read = kwargs.get('n_packets')
            for i in range(packets_to_read):
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
        report_performance(type(self).__name__, log, packet_count, sum_processing_time)

        # Data loaders only exists once per data source, therefore they are
        # suitable for tracking the overall number of packets processed. This
        # value will be reported by the main pipeline in the end.
        global_variables.global_pipeline_packet_count += packet_count


    def randomize_packets_per_concept(
        self,
        new_feature_stream: List[Generator[Dict[Any, Any], None, None]],
        n_subsets_per_concept: List[int],
        packets_per_subset: List[int], 
        labels: List[int]
        ) -> Tuple[Generator[Dict[Any, Any], None, None], List[int]]:
        
        """
        Randomly samples packets from each subdataset defined by new_feature_streams,
        grouping them into blocks based on n_subsets_per_concept, and shuffling samples
        within each block. Returns shuffled samples and their corresponding indices.

        Parameters:
        - new_feature_streams: List of generators producing packet samples from subdatasets.
        - n_subsets_per_concept: List of integers specifying how many generators to group
        together as subsets for sampling.
        - packets_per_subset: List of integers specifying the number of packets to sample
        from each subset.
        - labels: List of integers representing labels for each subset.

        Returns:
        - A tuple containing:
        - A generator producing the packet samples in a random order.
        - A list of integers representing the order of subdataset indices from which
            the samples were taken.
        """

        collected_samples = []
        collected_index = []
        
        subset_labels = iter(labels)  # Create an iterator over labels
        subset_packets = iter(packets_per_subset)  # Create an iterator over labels

        # Step 1: Collect samples from each subdataset and maintain the index vector
        for n_generators in n_subsets_per_concept:
            

            # Create vector with labels of the current n_generators being processed out of new_feature_stream
            index_vector = []
            for i in range(n_generators):
                current_subset_label = next(subset_labels)
                current_subset_packets = next(subset_packets)
                index_vector.extend([current_subset_label] * current_subset_packets)

            # Collect samples from current subset of generators
            subset_samples = []
            for stream in new_feature_stream[:n_generators]:
                subset_samples.extend(stream)
            
            # Remove processed generators from the list
            new_feature_stream = new_feature_stream[n_generators:]

            # Pair the samples with their indices and shuffle
            paired_samples = list(zip(subset_samples, index_vector))
            random.shuffle(paired_samples)

            # Unzip the shuffled pairs into separate lists
            shuffled_samples, shuffled_index = zip(*paired_samples) if paired_samples else ([], [])

            # Extend collected samples and index vector
            collected_samples.extend(shuffled_samples)
            collected_index.extend(shuffled_index)

        return (sample for sample in collected_samples), list(collected_index)


    @staticmethod
    def feature_signature() -> List[IFeature]:
        return [PacketFeature.CPP_FEATURE_STRING]
