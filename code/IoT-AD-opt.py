import argparse
import itertools
import json
import os
import time
from typing import List

from jinja2 import Template
from collections import deque
from collections import Counter

import optuna

import numpy as np
np.float = float
np.int = np.int32
np.bool = np.bool_


from river import anomaly, tree, linear_model
from river import preprocessing, stream, datasets
from river import model_selection, optim, bandit
from river import metrics, evaluate
from river import utils

# from scipy import integrate


from river.metrics import ConfusionMatrix, ROCAUC

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
            ### RMC 2024-0from river import stream5-21: the description above is concentrating an additional function in the feature extractors: data formatting of data from different sources.
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

        # Count occurrencfrom river import streames of each concept ID
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

        if configuration["MODEL"].get("interleave_samples", False):
            # Randomize and sample packets
            log.info(f"Interleaving order of samples (probably flow) from different datasets into a single random stream.")
            log.info(f"Number of subsets before each concept drift: {n_subsets_per_concept}. No. of packets per subset: {samples_per_subset}. Labels of all subsets, used for order randomization: {shuffling_labels}.")
            feature_stream, sample_order = loader.interleave_samples_per_concept(feature_streams, n_subsets_per_concept, samples_per_subset, shuffling_labels, max_flows_per_pick)


        elif configuration["MODEL"].get("randomize_samples", False):
            # Randomize and sample packets
            log.info(f"Randomizing order of samples (probably packets) from different datasets into a single random stream.")
            log.info(f"Number of subsets before each concept drift: {n_subsets_per_concept}. No. of packets per subset: {samples_per_subset}. Labels of all subsets, used for order randomization: {shuffling_labels}.")
            feature_stream, sample_order = loader.randomize_samples_per_concept(feature_streams, n_subsets_per_concept, samples_per_subset, shuffling_labels)
        
        else:
            def chain_generators(trafficstreams):
                for trafficstream in trafficstreams:
                    yield from trafficstream
        
            feature_stream = chain_generators(feature_streams)

        # print(samples_per_subset, sample_order)


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
        model_instance: IAnomalyDetectionModel = model_class(
            full_config_json=json.dumps(configuration, indent=4), **model_specification
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
        ###########################

        # 2024-10-08: IMPLEMENT SAVING SCHEME. Problem is: encoded_features is a structure with 2 elements: samples and ecoding.
        # The first is a numpy array of all elements of the sample (including ground_truth), while the second is a list of the encoded features.
        # I need to first merge them into the csv, then find a way to unmerge them and use in the predict method of the classes of ML models.





    #############################
    #### OPTIMIZE HYPERPARAMS ###
    #############################




    ################################ HARDCODED PREPARISON FOR RIVER (needs to change this preprocessing to the preprocessing class)
    
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
    # print("testing x")
    # print(x)
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
        'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
        'feature11', 'feature12']
    encoded_x = [dict(zip(feature_names, arr)) for arr in x]

    # # Create a generator for River
    # def river_generator():
    #     for features, target in zip(encoded_x, y):
    #         yield features, target
    
    riverdataset = stream.iter_array(x, y, feature_names=['x1', 'x2', 'x3', 'x4'])
    # count = 0
    # for x,y in riverdataset:
    #     count+count


    ############################################### LOGISTIC REGRESSION

    # # optimizer = optim.AdaMax()
    # model = (
    #     preprocessing.StandardScaler() |
    #     linear_model.LogisticRegression()
    # )
    # metric = metrics.F1()
    # evaluate.progressive_val_score(riverdataset, model, metric, print_every=100)

    ############################################### HOEFFDING

    ## RIVER OPT

    # optimizer = optim.Adam()
    # model = (        
    #     # preprocessing.StandardScaler() |
    #     tree.HoeffdingAdaptiveTreeClassifier(optimizer)
    # )
    # metric = metrics.Accuracy()
    # evaluate.progressive_val_score(riverdataset1, model, metric, print_every=100)
    # # evaluate.progressive_val_score(bikedataset, model, metric, print_every=500)


    

    opt_option = 3


    if opt_option == 1:

        model = (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, leaf_prediction='nba')
        )
        evaluate.progressive_val_score(
            dataset=riverdataset,
            model=model,
            metric=metrics.Accuracy(),
            print_every=500
        )

    elif opt_option == 2:
    
        
        model = (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, leaf_prediction='nba')
        )

        models = utils.expand_param_grid(model, 
        {
            'max_depth': [1, 2, 3],
            'tau': [0.01, 0.05, 0.1],
            'nb_threshold': [0, 1, 2]
        },
        {
            'optimizer': [
            (optim.SGD, {'lr': [.1, .01, .005]}),
                (optim.Adam, {'beta_1': [.01, .001], 'lr': [.1, .01, .001]}),
                (optim.Adam, {'beta_1': [.1], 'lr': [.001]}),
            ]
        }
        )

        # models = utils.expand_param_grid(model, {
        #         'max_depth': [1, 2, 3],
        #         'tau': [0.01, 0.05, 0.1],
        #         'nb_threshold': [0, 1, 2]
        #     }
        # )


        sh = model_selection.SuccessiveHalvingClassifier(
            models,
            metric=metrics.Accuracy(),
            budget=5000,
            eta=2,
            verbose=True
        )

        evaluate.progressive_val_score(
            dataset=riverdataset,
            model=sh,
            metric=metrics.Accuracy(),
            # print_every=500
        )

        print("Output SuccessiveHalving")
        print(sh.best_model)
        # sh.best_model gives you the best pipeline (StandardScaler | HoeffdingAdaptiveTreeClassifier)
        best_pipeline = sh.best_model

        # Access the HoeffdingAdaptiveTreeClassifier inside the pipeline
        best_classifier = best_pipeline['HoeffdingAdaptiveTreeClassifier']

        # # Now you can print its parameters
        print(f"Best Classifier Details: {best_classifier}")
        print(f"Grace Period: {best_classifier.grace_period}")
        print(f"Max Depth: {best_classifier.max_depth}")
        print(f"Tau: {best_classifier.tau}")
        print(f"Naive Bayes Threshold: {best_classifier.nb_threshold}")
        print(f"Split criterion: {best_classifier.split_criterion}")

    elif opt_option == 3:

        model = (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, leaf_prediction='nba')
        )

        models = utils.expand_param_grid(model,
            {
                'max_depth': [1, 2, 3],
                'tau': [0.01, 0.05, 0.1],
                'nb_threshold': [0, 1, 2],
                'optimizer': [
                    (optim.SGD),
                    (optim.AdaMax),
                    (optim.Adam),
                    (optim.AdaGrad),
                    (optim.AdaDelta),
                    (optim.AdaBound),
                    (optim.AMSGrad),
                    ]
            }
        )
        #         'optimizer': [
        #             (optim.SGD, {'lr': [.1, .01, .005]}),
        #             (optim.Adam, {'beta_1': [.01, .001], 'lr': [.1, .01, .001]}),
        #             (optim.Adam, {'beta_1': [.1], 'lr': [.001]}),
        #             ]
        #     }
        # )

        print(models[2])


        sh = model_selection.BanditClassifier(
            models,
            metric=metrics.Accuracy(),
            policy=bandit.EpsilonGreedy(
                epsilon=0.1,
                decay=0.001,
                burn_in=20,
                seed=42
            )
        )

        evaluate.progressive_val_score(
            dataset=riverdataset,
            model=sh,
            metric=metrics.Accuracy(),
            print_every=100
        )

        print("Output Bandit")
        print(sh.best_model)
        # sh.best_model gives you the best pipeline (StandardScaler | HoeffdingAdaptiveTreeClassifier)
        best_pipeline = sh.best_model

        # Access the HoeffdingAdaptiveTreeClassifier inside the pipeline
        best_classifier = best_pipeline['HoeffdingAdaptiveTreeClassifier']

        # # Now you can print its parameters
        print(f"Best Classifier Details: {best_classifier}")
        print(f"Grace Period: {best_classifier.grace_period}")
        print(f"Max Depth: {best_classifier.max_depth}")
        print(f"Tau: {best_classifier.tau}")
        print(f"Naive Bayes Threshold: {best_classifier.nb_threshold}")
        print(f"Split criterion: {best_classifier.split_criterion}")

        # To access the optimizer, assuming it is set as an attribute in the classifier
        if hasattr(best_classifier, 'optimizer'):
            optimizer = best_classifier.optimizer
            print(f"Optimizer Type: {type(optimizer).__name__}")
            
            if isinstance(optimizer, optim.SGD):
                print(f"Learning Rate (lr): {optimizer.lr}")
                
            elif isinstance(optimizer, optim.Adam):
                print(f"Learning Rate (lr): {optimizer.lr}")
                print(f"Beta_1: {optimizer.beta_1}")
                print(f"Beta_2: {optimizer.beta_2}")
                print(f"Eps: {optimizer.eps}")

        else:
            print("No optimizer found in the classifier.")
        

    # ### OPTUNA

    # # Define the objective function for optimization
    # def objective(trial):
    #     # Suggest hyperparameters
    #     grace_period = trial.suggest_int('grace_period', 50, 500)
    #     max_depth = trial.suggest_int('max_depth', 1, 100)
    #     split_criterion = trial.suggest_categorical('split_criterion', ['gini', 'info_gain', 'hellinger'])
    #     delta = trial.suggest_float('delta', 1e-8, 1e-1, log=True)
    #     tau = trial.suggest_float('tau', 0.01, 0.1)
    #     leaf_prediction = trial.suggest_categorical('leaf_prediction', ['mc', 'nb', 'nba'])
    #     nb_threshold = trial.suggest_int('nb_threshold', 0, 100)

    #     # Create the model
    #     model = tree.HoeffdingTreeClassifier(
    #         grace_period=grace_period,
    #         max_depth=max_depth,
    #         split_criterion=split_criterion,
    #         delta=delta,
    #         tau=tau,
    #         leaf_prediction=leaf_prediction,
    #         nb_threshold=nb_threshold
    #     )
        
    #     # Use a scaler if necessary
    #     model = preprocessing.StandardScaler() | model

    #     # Define the evaluation metric
    #     metric = metrics.F1()

    #     # # Perform cross-validation
    #     # for x, y in model_selection.iterative_train_test_split(riverdataset):
    #     #     model.learn_one(x, y)
    #     #     y_pred = model.predict_one(x)
    #     #     metric = metric.update(y, y_pred)
        
    #     # Evaluate the model
    #     metric = evaluate.progressive_val_score(riverdataset, model, metric)
    #     return metric

    # # Create a study
    # study = optuna.create_study(direction='maximize')

    # # Optimize the objective function
    # study.optimize(objective, n_trials=50)

    # # Output the best hyperparameters and the best score
    # print("Best hyperparameters:", study.best_params)
    # print("Best F1 Score:", study.best_value)

    ############################################### HALFSPACETREE

    ### RIVER OPT

    # # optimizer = optim.AdaBound()
    # model = (
    #     preprocessing.StandardScaler() |
    #     # anomaly.HalfSpaceTrees(optimizer)
    #     anomaly.HalfSpaceTrees(n_trees=7, height=10, window_size=50, seed=42)
    # )
    # metric = metrics.ROCAUC()


    # evaluate.progressive_val_score(riverdataset, model, metric, print_every=500)
    

    ### OPTUNA

    # # Define the objective function for optimization
    # def objective(trial):
    #     n_trees = trial.suggest_int('n_trees', 2, 10)
    #     height = trial.suggest_int('height', 2, 10)
    #     window_size = trial.suggest_int('window_size', 50, 100)
        
    #     model = anomaly.HalfSpaceTrees(seed=42, n_trees=n_trees, height=height, window_size=window_size)
    #     metric = metrics.ROCAUC()
        
    #     # Evaluate the model
    #     score = evaluate.progressive_val_score(riverdataset, model, metric)
    #     return score

    # # Perform optimization
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=20)

    # # Best hyperparameters
    # print(study.best_params)


    ############################################### PLOTTINGs

    # best_model = anomaly.HalfSpaceTrees(
    #     seed=42,
    #     n_trees=study.best_params['n_trees'],
    #     height=study.best_params['height'],
    #     window_size=study.best_params_['window_size']
    # )

    # # Initialize metrics
    # confusion_matrix = ConfusionMatrix()
    # roc_auc = ROCAUC()

    # # Reset the generator to evaluate the model on the dataset
    # riverdataset = stream.iter(river_generator(encoded_feature_generator))

    # # Evaluate the model and track predictions and true values
    # for features, target in riverdataset:
    #     y_pred = best_model.predict_one(features)
    #     confusion_matrix = confusion_matrix.update(target, y_pred)
    #     roc_auc = roc_auc.update(target, y_pred)

    # # Print confusion matrix and other metrics
    # print("Confusion Matrix:")
    # print(confusion_matrix)

    # # Calculate and print other metrics
    # print("Accuracy:", confusion_matrix.accuracy)
    # print("Precision:", confusion_matrix.precision)
    # print("Recall:", confusion_matrix.recall)
    # print("F1 Score:", confusion_matrix.f1)

    # # Collect data for ROC curve
    # y_true = []
    # y_scores = []

    # # Reset the generator again for ROC curve data collection
    # riverdataset = stream.iter(river_generator(encoded_feature_generator))

    # for features, target in riverdataset:
    #     y_true.append(target)
    #     y_scores.append(best_model.predict_proba_one(features)[1])  # Assuming binary classification

    # # Compute ROC curve
    # fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # roc_auc_score = auc(fpr, tpr)

    # # Plot ROC AUC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_score))
    # plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()

    
    # reporter_instance.end_processing()



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



if __name__ == "__main__":

    # Argument parser initialization.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str, required=True)
    parser.add_argument("--influx-token", type=str, required=False, default="")
    log.debug("Parsing arguments.")
    args = parser.parse_args()

    main(args.config_path, args.influx_token)

