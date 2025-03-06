import argparse
import itertools
import json
import os
import time
from typing import List

from jinja2 import Template
from collections import deque
from collections import Counter
from datetime import datetime

from river import anomaly, tree, linear_model, forest
from river import preprocessing, stream, datasets
from river import model_selection, optim, bandit
from river import metrics, evaluate
from river import utils


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

    ### TODO 2024-12-05: find the best place to parameterize seed_randomization (in the config file) ###
    seed_randomization = 42

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




    #######################################################################################################################
    #######################################################################################################################
    ### SECTION OF DATA PROCESSING



    ##################
    ### DATA INPUT ###
    ##################

    if configuration["MODEL"]["data_format"] == "raw_pcap":

        ##########################
        ### DATA FROM RAW PCAP ###
        ##########################

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
            ###                 Data formatting and feature extraction are different functions. The former serves the purpose described above. The latter selects different
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


        ###########################################
        ### PREPARING SIMULATION OF STREAM DATA ###
        ###########################################

        # Count occurrences of each concept ID
        concept_counts = Counter(concept_id_per_subset)
        # Extract the counts in the order of appearance in concept_id_per_subset
        n_subsets_per_concept = [concept_counts[concept_id] for concept_id in sorted(concept_counts)]
        max_flows_per_pick = 3

        samples_per_subset = []

        copy_streams = []
        for generator in feature_streams:
            iter1, generator  = itertools.tee(generator)
            copy_streams.append(generator)
            count = 0    
            count = sum(1 for _ in iter1)
            samples_per_subset.append(count)
        
        feature_streams = copy_streams

        



        # Important step of streaming-data simulation, where samples are either randomized for packet-based feature extraction or interleaved for flow-based feature extraction.
        # Default loop is False for both in case neither was defined in the configuration file

        print(f"### TESTING LABELS {shuffling_labels} and samples per subset {samples_per_subset}")

        sample_order = None

        if configuration["MODEL"].get("interleave_samples", False):
            # Randomize and sample packets
            log.info(f"Interleaving order of samples (probably flow) from different datasets into a single random stream.")
            log.info(f"Number of subsets before each concept drift: {n_subsets_per_concept}. No. of packets per subset: {samples_per_subset}. Labels of all subsets, used for order randomization: {shuffling_labels}.")
            feature_stream, sample_order = loader.interleave_samples_per_concept(feature_streams, n_subsets_per_concept, samples_per_subset, shuffling_labels, max_flows_per_pick, seed_randomization)

        elif configuration["MODEL"].get("randomize_samples", False):
            # Randomize and sample packets
            log.info(f"Randomizing order of samples from different datasets into a single random stream.")
            log.info(f"Number of subsets before each concept drift: {n_subsets_per_concept}. No. of packets per subset: {samples_per_subset}. Labels of all subsets, used for order randomization: {shuffling_labels}.")
            feature_stream, sample_order = loader.randomize_samples_per_concept(feature_streams, n_subsets_per_concept, samples_per_subset, shuffling_labels, seed_randomization)
        
        else:
            def chain_generators(streams):
                for stream in streams:
                    yield from stream
        
            feature_stream = chain_generators(feature_streams)

        print(samples_per_subset, sample_order)

        # Printing into files


        output_dir = 'configurations/zplots'
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_subsets_path = os.path.join(output_dir, f'{current_time}_{configuration["MODEL"]["model_name"]}_sample_subsets.txt')
        sample_order_path = os.path.join(output_dir, f'{current_time}_{configuration["MODEL"]["model_name"]}_sample_order.txt')
        
        with open(sample_subsets_path, "w") as file:
            file.write("Randomizing order of samples from different datasets into a single random stream.\n\n")
            file.write(f"Number of subsets before each concept drift: {n_subsets_per_concept}.\n")
            file.write(f"No. of packets per subset: {samples_per_subset}.\n")
            file.write(f"Order of subsets, before sample randomization concept by concept: {shuffling_labels}.\n")

        with open(sample_order_path, "w") as file:
            file.write(", ".join(map(str, sample_order)))


        # Create two copies of the iterator using tee
        iter1 = itertools.tee(feature_stream)
        count = 0
        for _ in iter1:
            count += 1
        # print(count)  # Output: 10



        #####################
        ### INIT MODEL(S) ###
        #####################

        # Initialize model class based on the component specification in the configuration.
        model_specification = configuration["MODEL"]
        model_name = model_specification["class"]
        model_class = globals()[model_name]

        # Extract model_param separately
        model_param = model_specification.pop("model_param", {})

        model_instance: IAnomalyDetectionModel = model_class(
            full_config_json=json.dumps(configuration, indent=4),
            **model_specification
            # **model_param  # Unpack model_param here
        )




        ################
        ### ENCODING ###
        ################

        # Initialize encoder class for the model. Encoders are model-specific to allow running multiple models simultaneously in the future, where each may require their own encoder instance.
        encoder_name = model_specification["encoder"]["class"]
        encoder_class = globals()[encoder_name]
        encoder_instance: IDataEncoder = encoder_class(
            **model_specification["encoder"]["kwargs"]
        )
        log.info("Encoding features.")

        # This moment is important for performance measurement because encoding is the first step
        # where features are actually processed. Until here, the generator data has not been consumed, so no data processing needed to take place).

        encoding_start = time.process_time_ns()
        encoded_feature_generator = encoder_instance.encode(feature_stream)

        # Sanity check - peek at the first sample, print its fields and encoded format.
        peeker, encoded_feature_generator = itertools.tee(encoded_feature_generator)
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



        ###########################
        #### SAVING DATA IN CSV ###

        # 2024-10-08: IMPLEMENT SAVING SCHEME. Problem is: encoded_features is a structure with 2 elements: samples and ecoding.
        # The first is a numpy array of all elements of the sample (including ground_truth), while the second is a list of the encoded features.
        # I need to first merge them into the csv, then find a way to unmerge them and use in the predict method of the classes of ML models.




    elif configuration["MODEL"]["data_format"] == "processed_csv":
        
        ################################
        ### DATA FROM PROCESSED CSV  ###

        print("Implement loading processed csv")



    #######################################################################################################################
    #######################################################################################################################
    ### SECTION OF ML TRAINING / TESTING


    ######################
    ### BATCH LEARNING ###
    ######################

    # ML models based on scikit-learn and data structure based on numpy arrays. (It is different for STREAM LEARNING)

    if model_specification["data_flow"] == "batch":


        ###########################
        ### Train model offline ###
        ###########################

        if model_specification["ml_task"] == "train":

            if model_specification["new_model"]:

            # perform training for the first time
                # Train the model via batch of data
                model_instance.train(
                    encoded_feature_generator, path_to_store=model_instance.store_file
                )
                
            else:

                # New model: instantiation handles new or load model by itself. Old: If models exists, always load
                # model_instance.load(model_instance.store_file)

                log.error("Batch ML does not incrementally improve already existing models!")
                exit(1)
            



        ##########################
        ### Test model offline ###
        ##########################
        
        if model_specification["ml_task"] == "test":

            #####################
            ### Init reporter ###
            #####################

            reporter_instances: List[IReporter] = []

            # Initialize reporter
            for output in configuration["OUTPUT"]:
                reporter_name = output["class"]
                reporter_class = globals()[reporter_name]
                reporter_instance = reporter_class(**output["kwargs"])
                # model_param = configuration["MODEL"].pop("model_param", {})  # Extract model_param from combined_kwargs
                # reporter_instance = reporter_class(model_param=model_param, **output["kwargs"])
                reporter_instances.append(reporter_instance)
        
            
            ##############################
            ### Learn / Test Whole-set ###
            ##############################

            if model_specification["sampling_rate"] == "whole_set":

                if model_specification["new_model"]:
            
                    # Evaluate testing dataset, sample by sample efficiently, using Python generator keyword "yield" instead of "return" (see inside model_instance.predict method)
                    for predicted_sample in model_instance.predict(encoded_feature_generator):
                        for reporter_instance in reporter_instances:
                            reporter_instance.report(predicted_sample)

        
                else:

                    # New model: instantiation handles new or load model by itself. Old: If models exists, always load
                    # model_instance.load(model_instance.store_file)
                    
                    # Evaluate testing dataset, sample by sample efficiently, using Python generator keyword "yield" instead of "return" (see inside model_instance.predict method)
                    for predicted_sample in model_instance.predict(encoded_feature_generator):
                        for reporter_instance in reporter_instances:
                            reporter_instance.report(predicted_sample)
        


            #####################################
            ### Learn / Test Sample-by-sample ###
            #####################################

            elif model_specification["sampling_rate"] == "incremental":

                x = []
                y = []

                for samples, encoding in encoded_feature_generator:
                    if isinstance(samples, list):
                        # Handle the list with multiple samples used together with
                        # xarray DataArray encodings.
                        for f in samples:
                            y.append(f["ground_truth"])
                        if len(x) == 0:
                            x = encoding
                        else:
                            x = numpy.concatenate((x, encoding), axis=0)
                    else:
                        y.append(samples["ground_truth"])
                        x.append(encoding[0])
                        

                y_pred, cumulative_accuracies = model_instance.evaluate(x, y)

                for reporter_instance in reporter_instances:
                    reporter_instance.set_model_name(configuration["MODEL"]["model_name"])
                for i in range(0, len(y)):
                    for reporter_instance in reporter_instances:
                        reporter_instance.report_eval(y[i], y_pred[i])


                ###################################
                ### SHUTDOWN REPORTERS (REMOTE) ###
                ###################################

                # if model_specification["sampling_rate"] != "incremental":

                # Reporters may require special shutdown steps, for example disconnecting from
                # remote database or printing summaries of the processing -- call the handle for
                # each reporter.

                for reporter_instance in reporter_instances:
                    reporter_instance.end_processing()

                if not model_specification["skip_saving_model"]:
                    model_instance._save_model()




    #######################################################################################################################



    #######################
    ### STREAM LEARNING ###
    #######################

    # ML models based on River and data structure based on dictionaries. (It is different for BATCH LEARNING!)

    elif model_specification["data_flow"] == "stream":

        #############
        ### Train ###
        #############
        
        if model_specification["ml_task"] == "train":

            if model_specification["new_model"]:

                model_instance.train(
                    encoded_feature_generator, path_to_store=model_instance.store_file
                )
                
            else:

                # CORRECTION: instantiation handles new or load model by itself. OLD: If models exists, always load
                # model_instance.load(model_instance.store_file)
                
                model_instance.train(
                    encoded_feature_generator, path_to_store=model_instance.store_file
                )


        ############
        ### Test ###
        ############
        
        elif model_specification["ml_task"] == "test":

            #####################
            ### Init reporter ###
            #####################

            # if model_specification["sampling_rate"] != "incremental":
            reporter_instances: List[IReporter] = []

            # Initialize reporter
            for output in configuration["OUTPUT"]:
                reporter_name = output["class"]
                reporter_class = globals()[reporter_name]
                reporter_instance = reporter_class(**output["kwargs"])
                # model_param = configuration["MODEL"].pop("model_param", {})  # Extract model_param from combined_kwargs
                # reporter_instance = reporter_class(model_param=model_param, **output["kwargs"])
                reporter_instances.append(reporter_instance)
    
            
            ##############################
            ### Learn / Test Whole-set ###
            ##############################

            if model_specification["sampling_rate"] == "whole_set":

                if model_specification["new_model"]:
            
                    # Evaluate testing dataset, sample by sample efficiently, using Python generator keyword "yield" instead of "return" (see inside model_instance.predict method)
                    for predicted_sample in model_instance.predict(encoded_feature_generator):
                        for reporter_instance in reporter_instances:
                            reporter_instance.report(predicted_sample)

        
                else:

                    # CORRECTION: instantiation handles new or load model by itself. OLD: If models exists, always load
                    # model_instance.load(model_instance.store_file)
                    

                    # Evaluate testing dataset, sample by sample efficiently, using Python generator keyword "yield" instead of "return" (see inside model_instance.predict method)
                    for predicted_sample in model_instance.predict(encoded_feature_generator):
                        for reporter_instance in reporter_instances:
                            reporter_instance.report(predicted_sample)
                    # if data_source["loader"]["kwargs"].get("data_flow") == "stream":
                    #     streamtest()
        
            #####################################
            ### Learn / Test Sample-by-sample ###
            #####################################

            elif model_specification["sampling_rate"] == "incremental":

                x = []
                y = []

                for samples, encoding in encoded_feature_generator:
                    if isinstance(samples, list):
                        # Handle the list with multiple samples used together with
                        # xarray DataArray encodings.
                        for f in samples:
                            y.append(f["ground_truth"])
                        if len(x) == 0:
                            x = encoding
                        else:
                            x = numpy.concatenate((x, encoding), axis=0)
                    else:
                        y.append(samples["ground_truth"])
                        x.append(encoding[0])
                

                # # Necessary to scale samples, but River only works with dictionaries, so transforming
                feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                    'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                    'feature11', 'feature12']
                encoded_x = [dict(zip(feature_names, arr)) for arr in x]
                                
                riverdataset = stream.iter_array(x, y, feature_names=['x1', 'x2', 'x3', 'x4'])
                y_pred, cummulative_accuracies = model_instance.evaluate(riverdataset)

                for reporter_instance in reporter_instances:
                    reporter_instance.set_model_name(configuration["MODEL"]["model_name"])
                for i in range(0, len(y)):
                    for reporter_instance in reporter_instances:
                        reporter_instance.report_eval(y[i], y_pred[i])

                ###################################
                ### SHUTDOWN REPORTERS (REMOTE) ###
                ###################################

                # if model_specification["sampling_rate"] != "incremental":

                # Reporters may require special shutdown steps, for example disconnecting from
                # remote database or printing summaries of the processing -- call the handle for
                # each reporter.

                for reporter_instance in reporter_instances:
                    reporter_instance.end_processing()

                if not model_specification["skip_saving_model"]:
                    model_instance._save_model()



    ########################
    ### EVAL PERFORMANCE ###
    ########################

    # Function to convert nanoseconds to hours, minutes, seconds, and milliseconds


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

    bandwith_path = os.path.join(output_dir, f'{current_time}_{configuration["MODEL"]["model_name"]}_computation_costs.txt')

    with open(bandwith_path, "w") as file:
        file.write("---\nData volume and bandwidth:\n"
             f"  {global_variables.global_pipeline_packet_count} IP packets\n"
             f"  {global_variables.global_sum_ip_packet_sizes} bytes IP traffic\n"
             f"  {total_ethernet_bytes} bytes Ethernet traffic\n"
             f"  {round(from_processing_bandwidth, 2)} megabits/second "
             f"Ethernet traffic bandwidth from processing start\n"
             f"  {round(from_init_bandwidth, 2)} megabits/second "
             f"Ethernet traffic bandwidth from initialization start\n"
             f"  {round(total_pipeline_bandwidth, 2)} megabits/second "
             f"Ethernet traffic bandwidth for full pipeline\n"

        )



if __name__ == "__main__":

    # Argument parser initialization.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str, required=True)
    parser.add_argument("--influx-token", type=str, required=False, default="")
    log.debug("Parsing arguments.")
    args = parser.parse_args()

    main(args.config_path, args.influx_token)

