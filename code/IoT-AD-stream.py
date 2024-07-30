import argparse
import itertools
import json
import os
import time
from typing import List

from jinja2 import Template
from collections import deque
from collections import Counter


import common.global_variables as global_variables
from common.functions import report_performance, time_now, project_root, git_tag
from dataloaders import *
from models import *
from featextractors import *
from encoders import *
from reporting import *

from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


def main(args_config_path, args_influx_token):
    """
    Run the IoT anomaly detection pipeline based on a configuration file.
    """

    pipeline_execution_start = time.process_time_ns()
    log_time_tag = time_now()

    ####################
    ### CONFIG FILES ###
    ####################

    # Load configuration file that specifies pipeline components.
    config_path = os.path.abspath(args_config_path)
    config_file_name = os.path.basename(config_path).split('.')[0]
    assert os.path.exists(config_path), "Config file not found!"
    log.debug(f"Loading configuration from: {config_path}")
    with open(config_path) as config_file:
        if ".jinja" in config_path:
            # New functions for templating can be registered here.
            template = Template(config_file.read())
            template.globals["timestamp"] = log_time_tag
            template.globals["project_root"] = project_root()
            template.globals["git_tag"] = git_tag()
            template.globals["influx_token"] = args_influx_token
            template.globals["config_file_name"] = config_file_name
            configuration = json.loads(template.render())
        else:
            configuration = json.loads(config_path)
    if not configuration:
        log.error("Could not load configuration file!")
        exit(1)

    class_initialization_start = time.process_time_ns()

    ###############
    ### LOGGING ###
    ###############

    # Initialize file loggers.
    if "LOG" in configuration:
        for log_config in configuration["LOG"]:
            log_level = log_config.get("level", "DEBUG")
            # Default location is under the /logs directory in this repository.
            log_path = log_config.get(
                "path",
                os.path.join(
                    project_root(), "logs", "other", f"{log_time_tag}-log.txt"
                ),
            )
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            PipelineLogger.add_file_logger(log_level, log_path)

    # Re-logging the path because file-based logger was not initialized before.
    log.debug(f"Running configuration: {config_path}")

    ##################
    ### DATA INPUT ###
    ##################

    # List to store all feature streams (generators) from different datasets
    stream_concept_total = []
    new_feature_stream = []
    feature_streams = []
    packets_per_subset = []
    concept_id_per_subset = []
    shuffling_labels = []

    # Feature stream is a Python generator object: https://wiki.python.org/moin/Generators
    # It allows to process the samples memory-efficiently, avoiding the need to store all data in memory at the same time.
    feature_stream = itertools.chain([])

    # Initialize data loaders classes corresponding to each component under DATA_SOURCES in configuration.
    for data_source in configuration["DATA_SOURCES"]:

        ##################################
        ### Data loading (get samples) ###
        ##################################
        loader_name = data_source["loader"]["class"]
        loader_class = globals()[loader_name]
        log.info(f"Adding {loader_class.__name__} to pipeline.")
        loader: IDataLoader = loader_class(**data_source["loader"]["kwargs"])

        # Reading only part of the dataset in case the config file indicates so
        use_full_dataset = data_source["loader"]["kwargs"].get("use_full_dataset", True)
        if use_full_dataset:
            new_feature_stream = loader.get_samples(use_full_dataset=use_full_dataset)
        else:
            n_packets = data_source["loader"]["kwargs"].get("n_packets")
            packets_per_subset.append(n_packets)
            concept_id = data_source["loader"]["kwargs"].get("stream_concept_id")
            concept_id_per_subset.append(concept_id)
            shuffling_labels.append(len(shuffling_labels) + 1)
            new_feature_stream = loader.get_samples(use_full_dataset=use_full_dataset, n_packets=n_packets)

        ##########################
        ### Feature Extraction ###
        ##########################

        # Initialize feature extractors specific to the data sources. Allowing each data source
        # to specify its own feature extractor means data from different storage formats and
        # with different processing needs can be combined to train models or perform prediction.

        ### TODO: create a different submodule for data formatting when necessary
        ### RMC 2024-05-21: renamed module from preprocessor to featureextractor, because I will implement true data processing now
        ### RMC 2024-05-21: the description above is concentrating an additional function in the feature extractors: data formatting of data from different sources.
        ###                 Data formatting and feature extractionare different functions. The former serves the purpose described above. The latter selects different
        ###                 sets of features, and in the case of flows, aggregates packet features.

        for featextractor_specification in data_source["featextractors"]:
            featextractor_name = featextractor_specification["class"]
            featextractor_class = globals()[featextractor_name]
            log.info(f"Adding {featextractor_class.__name__} to pipeline.")
            featextractor: IFeatExtractor = featextractor_class(
                **featextractor_specification["kwargs"]
            )
            new_feature_stream = featextractor.extract(new_feature_stream)

        feature_streams.append(new_feature_stream)
        #feature_streams = itertools.chain(feature_streams, new_feature_stream)
        # print(type(new_feature_stream))
        # print(type(feature_streams))
    
    
    # Count occurrences of each concept ID
    concept_counts = Counter(concept_id_per_subset)
    # Extract the counts in the order of appearance in concept_id_per_subset
    n_subsets_per_concept = [concept_counts[concept_id] for concept_id in sorted(concept_counts)]

    # Randomize and sample packets
    log.info(f"Randomizing order of packets from different datasets into a single random stream.")
    log.info(f"Number of subsets before each concept drift: {n_subsets_per_concept}. No. of packets per subset: {packets_per_subset}. Labels of all subsets, used for order randomization: {shuffling_labels}.")
    feature_stream, sample_order = loader.randomize_packets_per_concept(feature_streams, n_subsets_per_concept, packets_per_subset, shuffling_labels)
    # print(type(feature_streams))
    # print(type(feature_stream))
    
    # Create two copies of the iterator using tee
    iter1 = itertools.tee(feature_stream)
    count = 0
    for _ in iter1:
        count += 1
    # print(count)  # Output: 10

    #feature_stream = itertools.chain(feature_stream, stream_concept_total)
   
    # If no model is specified, count the number of samples in the loaded data.
    # Just a convenience function, might be removed later.
    if len(configuration["MODEL"]) == 0:
        log.info("No model specified - counting input data points:")
        count = 0
        for _ in feature_stream:
            count += 1
        log.info(f"{count} elements.")
        exit(0)

    ####################################
    ### INIT MODEL(S) AND ENCODER(S) ###
    ####################################

    # Initialize model class based on the component specification in the configuration.
    model_specification = configuration["MODEL"]
    model_name = model_specification["class"]
    model_class = globals()[model_name]
    model_instance: IAnomalyDetectionModel = model_class(
        full_config_json=json.dumps(configuration, indent=4), **model_specification
    )

    # Initialize encoder class for the model. Encoders are model-specific to allow running multiple models simultaneously in the future, where each may require their own encoder instance.
    encoder_name = model_specification["encoder"]["class"]
    encoder_class = globals()[encoder_name]
    encoder_instance: IDataEncoder = encoder_class(
        **model_specification["encoder"]["kwargs"]
    )
    log.info("Encoding features.")

    ################
    ### ENCODING ###
    ################

    # This moment is important for performance measurement because encoding is the first step
    # where features are actually processed. Until here, the generator data has not been consumed, so no data processing needed to take place).

    encoding_start = time.process_time_ns()
    encoded_feature_generator = encoder_instance.encode(feature_stream)

    # Sanity check - peek at the first sample, print its fields and encoded format.
    peeker, encoded_feature_generator = itertools.tee(encoded_feature_generator)
    peeker1, encoded_feature_generator1 = itertools.tee(encoded_feature_generator)
    first_sample1 = next(peeker1)
    first_sample_data1, _ = first_sample1
    print(first_sample_data1)
    first_sample = next(peeker)
    if not first_sample:
        log.warning("No data in encoded feature stream!")
    elif len(first_sample) == 2:  # Assure sample matches the intended signature.
        log.debug("Features of the first sample:")
        first_sample_data, _ = first_sample
        if isinstance(first_sample_data, list):
            # Extract first sample from list as encoded by MultiSampleEncoder. Otherwise, the first_sample_data object is already a dict containing the features of a single sample.
            first_sample_data = first_sample_data[0]
        for k, v in first_sample_data.items():
            if k == "cpp_feature_string":
                v = v.rstrip()  # Remove the newline character from the cpp_feature_string item
            log.debug(f" | {k}: {v}")


    #############################
    ### TRAIN MODEL OFFLINE ###
    #############################

    if model_specification["ml_task"] == "train":
    #if model_specification["train"]:
    #if you want to perform training and testing in the same run, do the following:
    # 1) delete model_specification["ml_task"] == "train": (and equivalent line from training code block)
    # 2) uncomment line below
    # 3) delete flag "ml_task" from config file
    # 4) create 2 new boolean flags: "train" and "test"

        if model_specification["new_model"]:
    
            if model_specification["dataflow"] == "batch":
            # perform training for the first time
                # Train the model via batch of data
                model_instance.train(
                    encoded_feature_generator, path_to_store=model_instance.store_file
                )
            
            if model_specification["dataflow"] == "stream":
                model_instance.train(
                    encoded_feature_generator, path_to_store=model_instance.store_file
                )
            
        else:

            # CORRECTION: instantiation handles new or load model by itself. OLD: If models exists, always load
            # model_instance.load(model_instance.store_file)
            
            # Also, if models exists and flag "train" is true, the model will be incrementally improved on top of previous training
        
            if model_specification["dataflow"] == "batch":
                log.error("Batch ML does not incrementally improve already existing models!")
                exit(1)
        
            if model_specification["dataflow"] == "stream":
                model_instance.train(
                    encoded_feature_generator, path_to_store=model_instance.store_file
                )

    ####################################################
    ### TEST MODEL OFFLINE / TRAIN-TEST MODEL ONLINE ###
    ####################################################
    
    if model_specification["ml_task"] == "test":
    #if model_specification["test"]:
    #if you want to perform training and testing in the same run, do the following:
    # 1) delete model_specification["ml_task"] == "test": (and equivalent line from training code block)
    # 2) uncomment line below
    # 3) delete flag "ml_task" from config file
    # 4) create 2 new boolean flags: "train" and "test"
    

        reporter_instances: List[IReporter] = []

        # Initialize reporter
        for output in configuration["OUTPUT"]:
            reporter_name = output["class"]
            reporter_class = globals()[reporter_name]
            reporter_instance = reporter_class(**output["kwargs"])
            reporter_instances.append(reporter_instance)

        if model_specification["new_model"]:
    
            # Evaluate testing dataset, sample by sample efficiently, using Python generator keyword "yield" instead of "return" (see inside model_instance.predict method)
            for predicted_sample in model_instance.predict(encoded_feature_generator):
                for reporter_instance in reporter_instances:
                    reporter_instance.report(predicted_sample)

           # TODO: substitute lines below the if above by the commented lines below, to differentiate between batch and stream testing in case you need it.
            # There will be one of the blocks below for each testing.
            # if data_source["loader"]["kwargs"].get("dataflow") == "batch":
            #     test()
            # if data_source["loader"]["kwargs"].get("dataflow") == "stream":
            #     streamtest()

 
        else:

            # CORRECTION: instantiation handles new or load model by itself. OLD: If models exists, always load
            # model_instance.load(model_instance.store_file)
            
            # TODO: substitute lines below the else above by the commented lines below, to differentiate between batch and stream testing in case you need it.
            # There will be one of the blocks below for each testing.
            # if data_source["loader"]["kwargs"].get("dataflow") == "batch":
            #     test()
            # if data_source["loader"]["kwargs"].get("dataflow") == "stream":
            #     streamtest()
 

            # Evaluate testing dataset, sample by sample efficiently, using Python generator keyword "yield" instead of "return" (see inside model_instance.predict method)
            for predicted_sample in model_instance.predict(encoded_feature_generator):
                for reporter_instance in reporter_instances:
                    reporter_instance.report(predicted_sample)

        
        print("cheguei aqui 2")
        # Reporters may require special shutdown steps, for example disconnecting from
        # remote database or printing summaries of the processing -- call the handle for
        # each reporter.
        for reporter_instance in reporter_instances:
            reporter_instance.end_processing()
            


    ########################
    ### EVAL PERFORMANCE ###
    ########################

    pipeline_stopping_time = time.process_time_ns()
    full_pipeline_time = pipeline_stopping_time - pipeline_execution_start
    time_from_initialization = pipeline_stopping_time - class_initialization_start
    time_from_processing = pipeline_stopping_time - encoding_start

    report_performance(
        "FullPipeline",
        log,
        global_variables.global_pipeline_packet_count,
        full_pipeline_time,
    )
    report_performance(
        "FromInitializationStart",
        log,
        global_variables.global_pipeline_packet_count,
        time_from_initialization,
    )
    report_performance(
        "FromProcessingStart",
        log,
        global_variables.global_pipeline_packet_count,
        time_from_processing,
    )

    # See Table 1 at:
    # https://sec.cloudapps.cisco.com/security/center/resources/network_performance_metrics.html
    total_ethernet_bytes = global_variables.global_sum_ip_packet_sizes + global_variables.global_pipeline_packet_count * 38
    total_pipeline_bandwidth = (total_ethernet_bytes * 8 / 1000000) / (full_pipeline_time / 1000000000)
    from_init_bandwidth = (total_ethernet_bytes * 8 / 1000000) / (time_from_initialization / 1000000000)
    from_processing_bandwidth = (total_ethernet_bytes * 8 / 1000000) / (time_from_processing / 1000000000)
    log.info("---\nData volume and bandwidth:\n"
             f"  {global_variables.global_pipeline_packet_count} IP packets\n"
             f"  {global_variables.global_sum_ip_packet_sizes} bytes IP traffic\n"
             f"  {total_ethernet_bytes} bytes Ethernet traffic\n"
             f"  {round(total_pipeline_bandwidth, 2)} megabits/second "
             f"Ethernet traffic bandwidth for full pipeline\n"
             f"  {round(from_init_bandwidth, 2)} megabits/second "
             f"Ethernet traffic bandwidth from initialization start\n"
             f"  {round(from_processing_bandwidth, 2)} megabits/second "
             f"Ethernet traffic bandwidth from processing start\n"
    )



if __name__ == "__main__":

    # Argument parser initialization.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str, required=True)
    parser.add_argument("--influx-token", type=str, required=False, default="")
    log.debug("Parsing arguments.")
    args = parser.parse_args()

    main(args.config_path, args.influx_token)

