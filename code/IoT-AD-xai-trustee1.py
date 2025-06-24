import itertools
import json
import os
import time
import pickle
import numpy as np
from typing import List

from jinja2 import Template

import common.global_variables as global_variables
from common.functions import report_performance, time_now, project_root, git_tag
from dataloaders import *
from models import *
from featextractors import *
from encoders import *
from reporting import *
from xai import *

from common.pipeline_logger import PipelineLogger

log = PipelineLogger.get_logger()


def main(args_config_path, args_influx_token):
    """
    Run the IoT anomaly detection pipeline based on a configuration file.
    """

    pipeline_execution_start = time.process_time_ns()
    log_time_tag = time_now()

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

    # Feature stream is a Python generator object: https://wiki.python.org/moin/Generators
    # It allows to process the samples memory-efficiently, avoiding the need to store all data in memory at the same time.
    feature_stream = itertools.chain([])

    # Initialize data loaders classes corresponding to each component under DATA_SOURCES in configuration.
    for data_source in configuration["DATA_SOURCES"]:
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
            featextractor: IFeatExtractor = featextractor_class(
                **featextractor_specification["kwargs"]
            )
            new_feature_stream = featextractor.extract(new_feature_stream)

        feature_stream = itertools.chain(feature_stream, new_feature_stream)

    # If no model is specified, count the number of samples in the loaded data.
    # Just a convenience function, might be removed later.
    if len(configuration["MODEL"]) == 0:
        log.info("No model specified - counting input data points:")
        count = 0
        for _ in feature_stream:
            count += 1
        log.info(f"{count} elements.")
        exit(0)

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
            log.debug(f" | {k}: {v}")

    if model_specification["train_new_model"]:
        # Train the model.
        model_instance.train(
            encoded_feature_generator, path_to_store=model_instance.store_file
        )

        # Save trained data for XAI 
        if "XAI" in configuration and configuration["XAI"]:
            with open(os.path.join(os.path.dirname(model_instance.store_file),
                                   model_instance.model_name + "_trained-data.pickle"), 'wb') as xai_file:
                pickle.dump(model_instance.get_training_data(), xai_file)

    else:
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

        # Save trained data for XAI
        if "XAI" in configuration and configuration["XAI"]:
            with open(os.path.join(os.path.dirname(model_instance.store_file),
                                   model_instance.model_name + "_test-data.pickle"), 'wb') as xai_file:
                pickle.dump(model_instance.get_test_data(), xai_file)

    # XAI starts
    if "XAI" in configuration and not model_specification["train_new_model"]:
        for xai_config in configuration["XAI"]:
            save_path = xai_config.get("path", None)
            if not save_path:
                log.error("The path to save XAI results was not specified in the configuration file")
                continue

            # Creates the directories if they do not exist
            os.makedirs(save_path, exist_ok=True)
            # Replace : with "" to avoid errors with filename formatting
            log_time_tag = log_time_tag.replace(":", "")
            # File with trained data is opened
            if os.path.exists(os.path.join(os.path.dirname(model_instance.store_file),
                                           model_instance.model_name + "_trained-data.pickle")):
                with open(os.path.join(os.path.dirname(model_instance.store_file),
                                       model_instance.model_name + "_trained-data.pickle"), 'rb') as xai_file:
                    trained_data = pickle.load(xai_file)

            # Formatting the features for XAI models
            extracted_features = [feature.split('.')[-1] for feature in model_instance.get_labels_name()]

            if os.path.exists(os.path.join(os.path.dirname(model_instance.store_file), "multiclass-labels.json")):
                with open(os.path.join(os.path.dirname(model_instance.store_file), "multiclass-labels.json"),
                          'rb') as xai_file:
                    class_names = list(json.load(xai_file).values())
            else:
                class_names = ['benign', 'anomalous']

            # File with test data is opened
            if os.path.exists(os.path.join(os.path.dirname(model_instance.store_file),
                                           model_instance.model_name + "_test-data.pickle")):
                with open(os.path.join(os.path.dirname(model_instance.store_file),
                                       model_instance.model_name + "_test-data.pickle"), 'rb') as xai_file:
                    test_data = pickle.load(xai_file)

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
            number_of_samples = xai_config.get("number_of_samples", None)
            if 'trustee' in xai_algo:
                # test_data = model_instance.get_test_data()
                getattr(model_instance, f"explain_with_{xai_algo}")(
                    X_train=trained_data['X'],
                    y_train=trained_data['y'],
                    X_test=test_data['X'],
                    y_test=test_data['y'],
                    y_predict=test_data['y_predict'],
                    class_names=class_names,
                    **xai_config
                )
            else:
                feature_mean_importance = getattr(model_instance, f"explain_with_{xai_algo}")(
                    train_data=trained_data,
                    test_data=test_data,
                    number_of_samples=number_of_samples
                )

            xai_time_stop = time.process_time_ns()

            if 'trustee' not in xai_algo:
                feature_importance_mapping = dict(zip(extracted_features, feature_mean_importance))
                feature_importance_sorted_dict = {k: v for k, v in
                                                  sorted(feature_importance_mapping.items(), key=lambda item: item[1])}

                log.info(f"FEATURE IMPORTANCE BY {xai_algo} ALGORITHM:")
                for key, value in feature_importance_sorted_dict.items():
                    log.info(f"{key} : {value:6.4f}")

            log.info(f"XAI - {xai_algo} finished after {xai_time_stop - xai_time_start} ns")

            # # SHAP
            # if xai_config.get("type", "").lower() == "shap":
            #     log.info(f"XAI - Model SHAP starts....")
            #     shpa_instance.explainer(model_instance, trained_data, extracted_features, class_names).savefig(
            #         os.path.join(save_path, f"{log_time_tag}-SHAP.png"))
            #     log.info(f"XAI - Model SHAP finished....")
            #     log.info(f"Results in {save_path}/{log_time_tag}-SHAP.png")
            # # LIME
            # elif xai_config.get("type", "").lower() == "lime":
            #     log.info(f"XAI - Model LIME starts....")
            #     lime_instance.explainer(trained_data, extracted_features, class_names)
            #
            #     log.info("****************** LIME: list of weighted features ***********************")
            #     for i, data in enumerate(model_instance.get_test_data()):
            #         array_values = data.values
            #         # If array_values has more than one row, take a single row for LIME
            #         if array_values.ndim == 2 and array_values.shape[0] > 1:
            #             test_row = array_values[0]
            #         else:
            #             test_row = array_values
            #         # Make sure test_row is a row
            #         test_row = np.squeeze(test_row)
            #
            #         def predict_fn(data):
            #             return model_instance.predict_proba(data)
            #
            #         lime_data = lime_instance.explainIntance(test_row, predict_fn, extracted_features).as_html()
            #         log.info(lime_instance.explainIntance(test_row, predict_fn, extracted_features).as_list())
            #
            #     with open(os.path.join(save_path, f"{log_time_tag}-LIME.html"), 'w') as f:
            #         f.write(lime_data)
            #
            #     log.info(f"XAI - Model LIME finished....")
            #     log.info(f"Results in {save_path}/{log_time_tag}-LIME.png")
            #
            # else:
            #     log.error("XAI type not supported")

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
