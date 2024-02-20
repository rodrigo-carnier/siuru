import subprocess
import time
from typing import List, Generator, Dict, Any

import common.global_variables as global_variables
from common.functions import report_performance
from dataloaders.IDataLoader import IDataLoader
from common.features import IFeature, PacketFeature

from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


class PacketSnifferLoop(IDataLoader):

    def __init__(self, network_interface: str, packet_processor_path: str, **kwargs):
        super().__init__(**kwargs)
        self.net_interface = network_interface
        self.preprocessor_path = packet_processor_path
        self.process = None  # Initialize process as an instance variable
        log.info(f"[{ type(self).__name__ }] Reading from network interface: {self.net_interface}")

    def start_capture(self):
        # Init sniffer using pcap_feature_extractor
        pcap_call = [self.preprocessor_path, "stream-device", self.net_interface]

        log.info(f"[PcapSniffer] Processing read interface {self.net_interface}")

        self.process = subprocess.Popen(
            pcap_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )
        log.info("Started capture")

        return None

    def stop_capture(self):
        if self.process is not None and self.process.poll() is None:
            # Terminate the subprocess if it is still running
            self.process.terminate()
            self.process.wait()  # Wait for the subprocess to finish

        self.process = None # To be explicit about termination

    def get_samples(self) -> Generator[Dict[IFeature, Any], None, None]:
        # Call stream_device directly and receive the captured packets
        log.info("Getting the samples!")
        captured_packets = self.stream_device()

        # Yield the captured packets
        for packet in captured_packets:
            packet_features = {PacketFeature.CPP_FEATURE_STRING: packet}
            yield packet_features

    def stream_device(self) -> List[str]:
        captured_packets = []

        while True:
            if self.process.poll() is not None:
                # print("Here is p.poll", process.poll())
                log.error(self.process.stdout.readlines())
                # print("Here is returncode", process.returncode)
                if self.process.returncode != 0:
                    raise RuntimeError(f"Sniffer feature extractor exited with error code {self.process.returncode}!")
            packet = self.process.stdout.readline()
            if packet:
                captured_packets.append(packet.strip())
            else:
                break


        report_performance(type(self).__name__, log, packet_count, sum_processing_time)

        # Data loaders only exist once per data source, therefore they are
        # suitable for tracking the overall number of packets processed. This
        # value will be reported by the main pipeline in the end.
        global_variables.global_pipeline_packet_count += packet_count

        return captured_packets

    @staticmethod
    def feature_signature() -> List[IFeature]:
        return [PacketFeature.CPP_FEATURE_STRING]
