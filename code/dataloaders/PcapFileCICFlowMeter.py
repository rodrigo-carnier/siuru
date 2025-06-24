import subprocess
import time
from typing import List, Tuple, Generator, Dict, Any
import cicflowmeter

import common.global_variables as global_variables
from common.functions import report_performance
from dataloaders.IDataLoader import IDataLoader
from common.features import IFeature, PacketFeature
from itertools import islice, chain
import random



from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()

import subprocess
import time
from typing import Generator, Dict, Any
from common.functions import report_performance
from dataloaders.IDataLoader import IDataLoader
from common.features import IFeature, PacketFeature
from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


class PcapFileCICFlowMeter(IDataLoader):
    def __init__(self, filepath: str, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        log.info(f"[{type(self).__name__}] Reading from file: {self.filepath}")

    def get_samples(self, **kwargs) -> Generator[Dict[IFeature, Any], None, None]:
        log.info(f"[{type(self).__name__}] Processing file: {self.filepath}")
        
        # Define the command-line call for CICFlowMeter
        cicflowmeter_call = [
            "cicflowmeter", "-f", self.filepath, "-c", "--fields", "all", "output.csv"
        ]
        
        sum_processing_time = 0
        packet_count = 0

        # Start subprocess to run CICFlowMeter command
        process = subprocess.Popen(
            cicflowmeter_call,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Assuming the output is in CSV format, we can read it line by line
        if kwargs.get('use_full_dataset', True):
            while True:
                start_time_ref = time.process_time_ns()

                # Check for process errors
                stderr_output = process.stderr.read()
                if stderr_output:
                    log.error(stderr_output)
                    raise RuntimeError(f"CICFlowMeter exited with error: {stderr_output}")

                # Read each line (representing a flow)
                flow_line = process.stdout.readline()
                if flow_line == "":  # End of file
                    break

                # Assuming each flow is a comma-separated value, split the line
                flow_features = flow_line.strip().split(",")
                print(flow_features)
                packet_features = {
                    PacketFeature.CPP_FEATURE_STRING: flow_features
                }

                sum_processing_time += time.process_time_ns() - start_time_ref
                yield packet_features
                packet_count += 1

        else:
            packets_to_read = kwargs.get('n_packets')
            for i in range(packets_to_read):
                start_time_ref = time.process_time_ns()

                # Read a flow line
                flow_line = process.stdout.readline()
                if flow_line == "":
                    break
                
                flow_features = flow_line.strip().split(",")
                print(flow_features)
                packet_features = {
                    PacketFeature.CPP_FEATURE_STRING: flow_features
                }

                sum_processing_time += time.process_time_ns() - start_time_ref
                yield packet_features
                packet_count += 1

        # Close subprocess when done
        process.stdout.close()
        process.stderr.close()
        process.wait()

        report_performance(type(self).__name__, log, packet_count, sum_processing_time)

        # Track the overall number of packets processed in the global pipeline
        global_variables.global_pipeline_packet_count += packet_count


    def randomize_samples_per_concept(
        self,
        new_feature_stream: List[Generator[Dict[Any, Any], None, None]],
        n_subsets_per_concept: List[int],
        packets_per_subset: List[int], 
        labels: List[int],
        seed: int = None
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
        - seed: Optional integer to seed the randomization for reproducibility.


        Returns:
        - A tuple containing:
        - A generator producing the packet samples in a random order.
        - A list of integers representing the order of subdataset indices from which
            the samples were taken.
        """

        # Seed the randomization
        if seed is not None:
            random.seed(seed)

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
            # print(current_subset_label, current_subset_packets, index_vector)

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


    def interleave_samples_per_concept(
        self,
        new_feature_stream: List[Generator[Dict[Any, Any], None, None]],
        n_subsets_per_concept: List[int],
        packets_per_subset: List[int], 
        labels: List[int],
        max_flows_per_pick: int,
        seed: int = None
        ) -> Tuple[Generator[Dict[Any, Any], None, None], List[int]]:


        """
        Sorts packets from each subdataset defined by new_feature_streams, interleaving
        flows in an ascending order across datasets while preserving the flow order within each dataset.
        Randomizes both the dataset selection and the number of flows picked from each dataset.
        Returns sorted samples and their corresponding indices.

        - A generator producing the packet samples in the sorted order.
        - A list of integers representing the order of subdataset indices from which
        the samples were taken.
        """

        # Seed the randomization
        if seed is not None:
            random.seed(seed)

        collected_samples = []
        collected_index = []
        
        subset_labels = iter(labels)  # Create an iterator over labels
        subset_packets = iter(packets_per_subset)  # Create an iterator over labels

        # Step 1: Collect samples from each subdataset and maintain the index vector
        for n_generators in n_subsets_per_concept:
            

            # Create vector with labels of the current n_generators being processed out of new_feature_stream
            index_vector = []
            dataset_streams = []
            for i in range(n_generators):
                current_subset_label = next(subset_labels)
                current_subset_packets = next(subset_packets)
                dataset_streams.append(new_feature_stream[i])  # Track the generators for this batch
                index_vector.extend([current_subset_label] * current_subset_packets)
            print(current_subset_label, current_subset_packets, index_vector)
            
            # Remove processed generators from the list
            new_feature_stream = new_feature_stream[n_generators:]

           # Step 2: Interleave flows in an ascending order across datasets
            interleaved_samples = []
            interleaved_index = []

            while any(dataset_streams):  # As long as there's data in any of the datasets
                # Randomly shuffle available datasets
                available_streams = [i for i, stream in enumerate(dataset_streams) if stream]
                
                if not available_streams:
                    break

                # Randomly choose a dataset to pick flows from
                dataset_idx = random.choice(available_streams)
                stream = dataset_streams[dataset_idx]
                
                try:
                    # Extract the next flow
                    flow = next(stream)
                    interleaved_samples.append(flow)
                    interleaved_index.append(current_subset_label-1+dataset_idx)
                except StopIteration:
                    # Remove the empty stream from the list
                    dataset_streams[dataset_idx] = None
                    break

                # # Pick a random number of flows between 1 and max_flows_per_pick
                # flows_to_pick = random.randint(1, max_flows_per_pick)

                # # Collect flows from the chosen dataset
                # for _ in range(flows_to_pick):
                    # try:
                    #     # Extract the next flow
                    #     flow = next(stream)
                    #     interleaved_samples.append(flow)
                    #     interleaved_index.append(index_vector[dataset_idx])
                    # except StopIteration:
                    #     # Remove the empty stream from the list
                    #     dataset_streams[dataset_idx] = None
                    #     break

            # Extend collected samples and index vector
            collected_samples.extend(interleaved_samples)
            collected_index.extend(interleaved_index)

        return (sample for sample in collected_samples), list(collected_index)


    @staticmethod
    def feature_signature() -> List[IFeature]:
        return [PacketFeature.CPP_FEATURE_STRING]
