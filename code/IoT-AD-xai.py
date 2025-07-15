import argparse
import itertools
import json
import os
import pickle
import time
from typing import List, Generator, Tuple, Dict, Any

import numpy as np
from jinja2 import Template
import xarray
import matplotlib


import common.global_variables as global_variables
from common.features import IFeature, PredictionField
from common.functions import report_performance, time_now, project_root, git_tag
from common.pipeline_logger import PipelineLogger
from dataloaders import *
from encoders import *
from models import *
from featextractors import *
from reporting import *

log = PipelineLogger.get_logger()


def reconstruct_encoded_generator(
    test_data: Dict[str, Any]
) -> Generator[Tuple[Dict[str, Any], xarray.DataArray], None, None]:
    """
    Convert test_data dict back to an EncodedSampleGenerator.
    """

    X = test_data["X"]
    y = test_data["y"]
    feature_names = test_data["feature_names"]

    for i in range(len(X)):
        # Simulate the Sample (dict containing at least ground truth)
        sample = {PredictionField.GROUND_TRUTH: y[i]}

        # Simulate the encoded xarray.DataArray with 2D shape
        encoded = xarray.DataArray(
            [X[i]],  # Make it 2D
            dims=["samples", "features"],
            coords={"features": feature_names}
        )

        yield (sample, encoded)

def collect_encoded_features(
    encoded_stream: Generator[Tuple[Dict[IFeature, Any], xarray.DataArray], None, None],
) -> np.ndarray:
    """
    Collects encoded feature arrays from any encoder output, ensuring a consistent
    (num_samples, num_features) shape.

    This function flattens both:
    - single-sample encodings from DefaultEncoder (shape: (1, num_features))
    - batch encodings from MultiSampleEncoder (shape: (batch_size, num_features))

    :param encoded_stream: Generator yielding (sample_dicts, xarray.DataArray)
    :return: A 2D NumPy array of shape (num_samples, num_features)
    """
    encoded_rows = []

    for sample_dicts, encoding in encoded_stream:
        # Ensure encoding is at least 2D (should always be (N, F))
        arr = encoding.values

        if arr.ndim == 1:
            # Unexpected 1D (e.g., shape (F,)), convert to (1, F)
            arr = arr[np.newaxis, :]
        elif arr.ndim == 2:
            # Normal case: shape (N, F)
            pass
        elif arr.ndim == 3 and arr.shape[0] == 1:
            # Edge case: (1, 1, F) → squeeze
            arr = arr.squeeze(axis=0)
        else:
            raise ValueError(f"Unexpected encoded shape: {arr.shape}")

        encoded_rows.append(arr)

    # Concatenate all into a 2D array
    return np.vstack(encoded_rows)

def parse_data_for_xai(data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None]):
    labels = []
    encoded_features = []

    data_prep_time = 0
    for samples, encoding in data:
        start = time.process_time_ns()
        if isinstance(samples, list):
            # Handle the list with multiple samples used together with
            # xarray DataArray encodings.
            for f in samples:
                labels.append(f[PredictionField.GROUND_TRUTH])
            if len(encoded_features) == 0:
                encoded_features = encoding
            else:
                encoded_features = np.concatenate(
                    (encoded_features, encoding), axis=0
                )
        else:
            labels.append(samples[PredictionField.GROUND_TRUTH])
            encoded_features.append(encoding[0])
        data_prep_time += time.process_time_ns() - start

    encoded_features = np.array(encoded_features)
    dataset = {
        'X': encoded_features,
        'y': np.array(labels)
    }
    return dataset


def initialize_file_loggers(log_configs, log_time_tag):
    for log_config in log_configs:
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


def initialize_data_source(data_source, feature_stream):
    loader_name = data_source["loader"]["class"]
    loader_class = globals()[loader_name]
    log.info(f"Adding {loader_class.__name__} to pipeline.")
    loader: IDataLoader = loader_class(**data_source["loader"]["kwargs"])
    new_feature_stream = loader.get_samples()

    # Initialize featextractors specific to the data sources. Allowing each data source to specify its own featextractor means data from different storage formats and with different processing needs can be combined to train models or perform prediction.
    for featextractor_specification in data_source["featextractors"]:
        featextractor_name = featextractor_specification["class"]
        featextractor_class = globals()[featextractor_name]
        log.info(f"Adding {featextractor_class.__name__} to pipeline.")
        featextractor: Ifeatextractor = featextractor_class(
            **featextractor_specification["kwargs"]
        )
        new_feature_stream = featextractor.extract(new_feature_stream)

    feature_stream = itertools.chain(feature_stream, new_feature_stream)
    return feature_stream


# Sanity check - peek at the first sample, print its fields and encoded format.
def sanity_check(encoded_feature_generator):
    peeker, encoded_feature_generator = itertools.tee(encoded_feature_generator)
    first_sample = next(peeker)
    feature_names = []
    if not first_sample:
        log.warning("No data in encoded feature stream!")
    elif len(first_sample) == 2:  # Assure sample matches the intended signature.
        log.debug("Features of the first sample:")
        first_sample_data, _ = first_sample
        if isinstance(first_sample_data, list):
            # Extract first sample from list as encoded by MultiSampleEncoder.
            # Otherwise, the first_sample_data object is already a dict containing the features of a single sample.
            first_sample_data = first_sample_data[0]
        for k, v in first_sample_data.items():
            log.debug(f" | {k}: {v}")
            feature_names.append(k)

    return feature_names, encoded_feature_generator


def main(args_config_path, args_influx_token):
    """
    Run the IoT anomaly detection pipeline based on a configuration file.
    """



    pipeline_execution_start = time.process_time_ns()
    log_time_tag = time_now()

    # LOAD CONFIGURATION FILE THAT SPECIFIES PIPELINE COMPONENTS.
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

    # INITIALIZE FILE LOGGERS.
    if "LOG" in configuration:
        initialize_file_loggers(configuration["LOG"], log_time_tag)

    # Re-logging the path because file-based logger was not initialized before.
    log.debug(f"Running configuration: {config_path}")

    # If no model is specified, count the number of samples in the loaded data.
    # Just a convenience function, might be removed later.
    if len(configuration["MODEL"]) == 0:
        log.info("No model specified - counting input data points:")
        # count = 0
        # for _ in feature_stream:
        #     count += 1
        # log.info(f"{count} elements.")
        exit(0)

    # # INITIALIZE MODEL CLASS BASED ON THE COMPONENT SPECIFICATION IN THE CONFIGURATION.
    # model_specification = configuration["MODEL"]
    # model_name = model_specification["class"]
    # model_class = globals()[model_name]
    # model_instance: IAnomalyDetectionModel = model_class(
    #     full_config_json=json.dumps(configuration, indent=4), **model_specification
    # )

    # Initialize model class based on the component specification in the configuration.
    model_specification = configuration["MODEL"]
    model_name = model_specification["class"]
    model_class = globals()[model_name]

    # Extract model_param separately
    model_param = model_specification.pop("model_param", {})

    # Create or load the ML black box model (class has inherited the loading)
    model_instance: IAnomalyDetectionModel = model_class(
        full_config_json=json.dumps(configuration, indent=4),
        **model_specification
        # **model_param  # Unpack model_param here
    )



    # INITIALIZE ENCODER CLASS FOR THE MODEL.
    # Encoders are model-specific to allow running multiple models simultaneously in the future,
    # where each may require their own encoder instance.
    encoder_name = model_specification["encoder"]["class"]
    encoder_class = globals()[encoder_name]
    encoder_instance: IDataEncoder = encoder_class(
        **model_specification["encoder"]["kwargs"]
    )
    log.info("Encoding features.")

    if model_specification["new_model"]:
        # INITIALIZE DATA LOADERS CLASSES CORRESPONDING TO EACH COMPONENT UNDER DATA_SOURCES IN CONFIGURATION. Feature
        # stream is a Python generator object: https://wiki.python.org/moin/Generators It allows to process the samples
        # memory-efficiently, avoiding the need to store all data in memory at the same time.
        feature_stream = itertools.chain([])

        for data_source in configuration["DATA_SOURCES"]:
            feature_stream = initialize_data_source(data_source, feature_stream)
        # This moment is important for performance measurement
        # because encoding is the first step where features are actually processed.
        # Until here, the generator data has not been consumed, so no data processing needed to take place).
        encoding_start = time.process_time_ns()

        encoded_feature_generator = encoder_instance.encode(feature_stream)
        _, encoded_feature_generator = sanity_check(encoded_feature_generator)

        # Train the model.
        model_instance.train(
            encoded_feature_generator, path_to_store=model_instance.store_file
        )

        if model_specification["save_data"]:
            dataset_suffix_name = model_specification["dataset_name"] if "dataset_name" in model_specification else "_trained-data"
            with open(
                    os.path.join(
                        os.path.dirname(model_instance.store_file),
                        model_instance.model_name + dataset_suffix_name + ".pickle"
                    ),
                    'wb'
            ) as xai_file:
                pickle.dump(model_instance.get_training_data(), xai_file)

    elif model_specification["evaluate_model"] is None or model_specification["evaluate_model"]:
        # INITIALIZE DATA LOADERS CLASSES CORRESPONDING TO EACH COMPONENT UNDER DATA_SOURCES IN CONFIGURATION. Feature
        # stream is a Python generator object: https://wiki.python.org/moin/Generators It allows to process the samples
        # memory-efficiently, avoiding the need to store all data in memory at the same time.
        feature_stream = itertools.chain([])

        for data_source in configuration["DATA_SOURCES"]:
            feature_stream = initialize_data_source(data_source, feature_stream)
        # This moment is important for performance measurement
        # because encoding is the first step where features are actually processed.
        # Until here, the generator data has not been consumed, so no data processing needed to take place).
        encoding_start = time.process_time_ns()

        encoded_feature_generator = encoder_instance.encode(feature_stream)
        _, encoded_feature_generator = sanity_check(encoded_feature_generator)

        # Prediction time!
        reporter_instances: List[IReporter] = []

        for output in configuration["OUTPUT"]:
            reporter_name = output["class"]
            reporter_class = globals()[reporter_name]
            reporter_instance = reporter_class(**output["kwargs"])
            reporter_instances.append(reporter_instance)

        for predicted_sample in model_instance.predict(encoded_feature_generator):
            for reporter_instance in reporter_instances:
                reporter_instance.report(predicted_sample)

        # Reporters may require special shutdown steps, for example disconnecting from
        # remote database or printing summaries of the processing -- call the handle for
        # each reporter.
        for reporter_instance in reporter_instances:
            reporter_instance.end_processing()

        if model_specification["save_data"]:
            dataset_suffix_name = model_specification["dataset_name"] if "dataset_name" in model_specification else "_test-data"
            with open(
                    os.path.join(
                        os.path.dirname(model_instance.store_file),
                        model_instance.model_name + dataset_suffix_name + ".pickle"
                    ),
                    'wb'
            ) as xai_file:
                # rodrigotest = model_instance.get_test_data()
                # # DEBUG PRINTS
                # print("=== DEBUG: Dumping rodrigotest ===")
                # if isinstance(rodrigotest, dict):
                #     X = rodrigotest.get('X')
                #     y = rodrigotest.get('y')
                #     print(" - X type:", type(X))
                #     try:
                #         print(" - X shape:", np.array(X).shape)
                #         if len(X) > 0:
                #             print(" - X[0] shape:", np.array(X[0]).shape)
                #     except Exception as e:
                #         print(" - Error checking X shape:", e)

                #     print(" - y type:", type(y))
                #     try:
                #         print(" - y shape:", np.array(y).shape)
                #         print(" - y length:", len(y))
                #     except Exception as e:
                #         print(" - Error checking y shape:", e)
                # else:
                #     print(" - rodrigotest is not a dict:", type(rodrigotest))
                # pickle.dump(rodrigotest, xai_file)
                pickle.dump(model_instance.get_test_data(), xai_file)


    # XAI starts
    if "XAI" in configuration:

        matplotlib.rcParams['font.family'] = 'DejaVu Serif'  # Or another installed serif font
        matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
        
        full_xai_config = configuration["XAI"]

        encoding_start = time.process_time_ns()

        extracted_features = []

        # Get train data
        train_data_path = full_xai_config.get("train_data_path", None)
        train_data_raw_sources = full_xai_config.get("train_data_raw_sources", [])

        if len(train_data_raw_sources) > 0:
            feature_stream = itertools.chain([])
            for data_source in train_data_raw_sources:
                feature_stream = initialize_data_source(data_source, feature_stream)
            encoded_feature_generator = encoder_instance.encode(feature_stream)

            # extracted_features, encoded_feature_generator = sanity_check(encoded_feature_generator)
            peeker, encoded_feature_generator = itertools.tee(encoded_feature_generator)
            for sample, encoding in peeker:
                extracted_features = np.array(encoding.features)
                break

            trained_data = parse_data_for_xai(encoded_feature_generator)
            print("DEBUG trained_data['X'] shape:", np.array(trained_data['X']).shape)
            print("DEBUG trained_data['y'] shape:", np.array(trained_data['y']).shape)


        else:
            if not train_data_path:
                train_data_path = os.path.join(os.path.dirname(model_instance.store_file),
                                               model_instance.model_name + "_trained-data.pickle")

            # File with trained data is opened
            if os.path.exists(train_data_path):
                with open(train_data_path, 'rb') as xai_file:
                    trained_data = pickle.load(xai_file)
                    print("=== DEBUG: Just loaded train_data from pickle ===")

            else:
                trained_data = {'X': [], 'y': [], 'feature_names': []}

            extracted_features = trained_data["feature_names"]

        # Get test data
        test_data_path = full_xai_config.get("test_data_path", None)
        test_data_raw_sources = full_xai_config.get("test_data_raw_sources", [])

        if len(test_data_raw_sources) > 0:
            #RMC
            print("######### AAAAAAHHHHH HUNG WHAT IS THIS BUG")
            feature_stream = itertools.chain([])
            for data_source in test_data_raw_sources:
                feature_stream = initialize_data_source(data_source, feature_stream)
            encoded_feature_generator = encoder_instance.encode(feature_stream)

            peeker, encoded_feature_generator = itertools.tee(encoded_feature_generator)
            for sample, encoding in peeker:
                extracted_features = np.array(encoding.features)
                break
            test_data = parse_data_for_xai(encoded_feature_generator)
            print("DEBUG test_data['X'] shape:", np.array(test_data['X']).shape)
            print("DEBUG test_data['y'] shape:", np.array(test_data['y']).shape)
        else:
            if not test_data_path:
                test_data_path = os.path.join(os.path.dirname(model_instance.store_file),
                                              model_instance.model_name + "_test-data.pickle")
            # File with test data is opened
            if os.path.exists(test_data_path):
                with open(test_data_path, 'rb') as xai_file:
                    test_data = pickle.load(xai_file)
                    print("=== DEBUG: Just loaded test_data from pickle ===")
                    print(" - type(test_data):", type(test_data))
                    X = test_data.get('X')
                    y = test_data.get('y')
                    print(" - X type:", type(X))
                    print(" - X shape:", np.array(X).shape)
                    if len(np.array(X).shape) > 1:
                        print(" - X[0] shape:", np.array(X[0]).shape)
                    print(" - y type:", type(y))
                    print(" - y shape:", np.array(y).shape)
                    print(" - y length:", len(y))
                    print(f" - y={y}")
            else:
                test_data = {'X': [], 'y': []}

        # Formatting the features for XAI models
        # extracted_features = [feature.split('.')[-1] for feature in model_instance.get_labels_name()]

        # Get anomaly labels (names)
        if os.path.exists(os.path.join(os.path.dirname(model_instance.store_file), "multiclass-labels.json")):
            with open(os.path.join(os.path.dirname(model_instance.store_file), "multiclass-labels.json"),
                      'rb') as xai_file:
                class_names = list(json.load(xai_file).values())
        else:
            class_names = ['benign', 'anomalous']

        # Run each XAI algorithms
        for xai_config in full_xai_config["algorithms"]:
            save_path = xai_config.get("path", None)
            if not save_path:
                log.error("The path to save XAI results was not specified in the configuration file")
                continue

            # Creates the directories if they do not exist
            os.makedirs(save_path, exist_ok=True)
            # Replace : with "" to avoid errors with filename formatting
            log_time_tag = log_time_tag.replace(":", "")

            xai_algo = xai_config.get("type", "").lower()
            if not xai_algo:
                log.info("No XAI Algorithm specified")
                continue

            if not callable(getattr(model_instance, f"explain_with_{xai_algo}", None)):
                raise TypeError(
                    f"The specified XAI algorithm is not supported by model {model_name}!"
                )

            log.info(f"xai - {xai_algo} starts...")
            xai_time_start = time.process_time_ns()

            xai_algo_return_feature_importance = ('rf', 'lime', 'shap')

            
            # print(trained_data)
            # print(test_data)
            # print(type(trained_data))
            # print(type(test_data))

            # # Defensive check and conversion
            # if not isinstance(trained_data['X'], np.ndarray):
            #     try:
            #         # Assume it's a list of dicts
            #         trained_data['X'] = np.array([list(x.values()) for x in trained_data['X']])
            #         trained_data['y'] = np.array([list(x.values()) for x in trained_data['y']])
            #         test_data['X'] = np.array([list(x.values()) for x in test_data['X']])
            #         test_data['y'] = np.array([list(x.values()) for x in test_data['y']])
            #     except Exception as e:
            #         raise TypeError(f"X_train must be a NumPy array or a list of dicts. Got {type(trained_data['X'])}. Conversion failed: {e}")

            # model_instance.predict_method_name = "predict_one"


            # Debug / check shapes and types before calling explain_with_*
            print("# Features names")
            print(test_data['feature_names'])

            print("@@@ Methods called in 1: IoT-AD.py")
            print([m for m in dir(model_instance) if callable(getattr(model_instance, m))])

            # print("@@@ AAAAHHHHHHHHHHHHHH Testing caller")
            # print(trained_data['X'])
            # print(type(trained_data['X']))
            # for pred in model_instance.predict(trained_data):
            #     print(pred)
            print("X_train shape:", np.array(trained_data['X']).shape)
            print("y_train shape:", np.array(trained_data['y']).shape)
            print("X_test shape:", np.array(test_data['X']).shape)
            print("y_test shape:", np.array(test_data['y']).shape)
            print("Unique values in y_train:", np.unique(trained_data['y']))
            print("Unique values in y_test:", np.unique(test_data['y']))

            # # RMC debuggin why river does not predict correctly in trustee: testing model_instance inside XAI run before passing to main.
            # # RMC ML black box model is WORKING for River
            # reconstructed = reconstruct_encoded_generator(test_data)
            # for predicted_sample in model_instance.predict(reconstructed):
            #     print(predicted_sample)


            # RMC Optional: sanity check that X_test and y_test have consistent first dimension size
            assert np.array(test_data['X']).shape[0] == np.array(test_data['y']).shape[0], "X_test and y_test size mismatch!"


            if xai_algo not in xai_algo_return_feature_importance:
                getattr(model_instance, f"explain_with_{xai_algo}")(
                    X_train=np.array(trained_data['X']),
                    y_train=np.array(trained_data['y']),
                    X_test=np.array(test_data['X']),
                    y_test=np.array(test_data['y']),
                    class_names=class_names,
                    feature_names=extracted_features,
                    save_path=save_path,
                    # prediction_method_name="predict"
                    **xai_config
                )
            else:
                feature_mean_importance = getattr(model_instance, f"explain_with_{xai_algo}")(
                    train_data=trained_data['X'],
                    test_data=test_data['X'],
                    feature_names=extracted_features,
                    **xai_config
                )

            xai_time_stop = time.process_time_ns()

            if xai_algo in xai_algo_return_feature_importance:
                feature_importance_mapping = dict(zip(extracted_features, feature_mean_importance))
                feature_importance_sorted_dict = {k: v for k, v in
                                                  sorted(feature_importance_mapping.items(), key=lambda item: item[1])}

                log.info(f"FEATURE IMPORTANCE BY {xai_algo} ALGORITHM:")
                for key, value in feature_importance_sorted_dict.items():
                    log.info(f"{key} : {value:6.4f}")

            log.info(f"XAI - {xai_algo} finished after {xai_time_stop - xai_time_start} ns")


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
