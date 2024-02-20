import subprocess
import time
from typing import List, Dict, Any
import threading
from queue import Queue


import common.global_variables as global_variables
from common.functions import report_performance
from dataloaders.IDataLoader import IDataLoader
from common.features import IFeature, PacketFeature

from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()

class PacketSnifferQueue:
    def __init__(self, network_interface: str, packet_processor_path: str, **kwargs):
        super().__init__(**kwargs)
        self.net_interface = network_interface
        self.preprocessor_path = packet_processor_path
        self.packet_limit = 100
        
        # Initialize a queue to store captured packets
        self.packet_queue = Queue()

        log.info(f"[{ type(self).__name__ }] Reading from network interface: {self.net_interface}")

        

    def get_samples(
        self,
    ) -> Generator[Dict[IFeature, Any], None, None]:
        pcap_call = [self.preprocessor_path, "stream-device", self.net_interface]

        log.info(f"[PcapSniffer] Processing read interface {self.net_interface}")
        sum_processing_time = 0
        packet_count = 0

        process = subprocess.Popen(
            pcap_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )

        packet_count = 0

        while packet_count < self.packet_limit:
            packet_feature_string = process.stdout.readline().strip()

            if not packet_feature_string:
                break  # Break out of the loop if there is no more output

            packet_features = {
                PacketFeature.CPP_FEATURE_STRING: packet_feature_string
            }

            sum_processing_time += time.process_time_ns() - start_time_ref

            # Store the captured packet in the queue
            self.packet_queue.put(packet_features)
            packet_count += 1

        process.terminate()
        process.wait()

        # Continue processing the remaining packets in the queue
        while not self.packet_queue.empty():
            yield self.packet_queue.get()



        while True:
            start_time_ref = time.process_time_ns()
            if process.poll() and process.returncode:
                log.error(process.stdout.readlines())
                raise RuntimeError(f"Sniffer feature extractor exited with error code {process.returncode}!")

            # Check if the stop flag is set
            if stop_flag.is_set():
                break

            # Read one line at a time
            packet_feature_string = process.stdout.readline().strip()

            if not packet_feature_string:
                break  # Break out of the loop if there is no more output

            packet_features = {
                PacketFeature.CPP_FEATURE_STRING: packet_feature_string
            }
            sum_processing_time += time.process_time_ns() - start_time_ref

            # Store the captured packet in the queue
            self.packet_queue.put(packet_features)
            packet_count += 1

        # Stop the capturing process
        process.terminate()
        process.wait()

        # Wait for the user input thread to finish
        user_input_thread.join()

        report_performance(type(self).__name__, log, packet_count, sum_processing_time)
        global_variables.global_pipeline_packet_count += packet_count

        # Continue processing the remaining packets in the queue
        while not self.packet_queue.empty():
            yield self.packet_queue.get()

    
        # RMC: next two lines -> where is this reported? how should I
        # change this for the sniffing in loop?
        report_performance(type(self).__name__, log, packet_count, sum_processing_time)

        # Data loaders only exist once per data source, therefore they are
        # suitable for tracking the overall number of packets processed. This
        # value will be reported by the main pipeline in the end.
        global_variables.global_pipeline_packet_count += packet_count

    @staticmethod
    def feature_signature() -> List[IFeature]:
        return [PacketFeature.CPP_FEATURE_STRING]
