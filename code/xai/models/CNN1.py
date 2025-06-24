import time
from typing import Generator, Any, Dict, Tuple

import os
import numpy
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv1D
from tensorflow.keras.layers import MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from joblib import dump, load
import shap
import lime
from lime import lime_tabular
import matplotlib.cm as cm
from omnixai.explainers.vision.specific.gradcam import GradCAM


from common.features import IFeature, PredictionField, EncodedSampleGenerator, SampleGenerator
from common.functions import report_performance
from common.pipeline_logger import PipelineLogger
from models.IAnomalyDetectionModel import IAnomalyDetectionModel
from xai.ITrusteeExplainableModel import ITrusteeExplainableModel

log = PipelineLogger.get_logger()


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


class CNNModel(IAnomalyDetectionModel, ITrusteeExplainableModel):
    def __init__(
            self,
            model_name,
            train_new_model=True,
            skip_saving_model=False,
            model_storage_base_path=None,
            model_relative_path=None,
            **kwargs,
    ):
        self.model_instance = None
        self.scaler = StandardScaler()
        # self.encoded_features = []
        self.train_data = {}
        self.feature_names = []
        # self.data_set = []
        self.test_data = {}

        super().__init__(
            model_name,
            train_new_model=train_new_model,
            skip_saving_model=skip_saving_model,
            model_storage_base_path=model_storage_base_path,
            model_relative_path=model_relative_path,
            **kwargs,
        )

    def build_model(self):
        filters = 64
        kernel_size = 3
        regularizer = "l2"
        strides = 1
        max_pool = 2
        dropout = 0.25

        model = Sequential(name=self.model_name)
        model.add(BatchNormalization())
        model.add(
            Conv1D(filters, kernel_size, strides, kernel_regularizer=regularizer, activation='relu', name='conv0'))
        model.add(MaxPooling1D(pool_size=max_pool, name='mp1'))
        model.add(Flatten())
        model.add(Dropout(dropout))
        model.add(Dense(256, activation='relu', name='fc1'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout * 2))
        model.add(Dense(1, activation='sigmoid', name='fc2'))
        return model

    def train(
            self,
            data: Generator[Tuple[Dict[IFeature, Any], np.ndarray], None, None],
            **kwargs,
    ):
        log.info("Training a CNN classifier.")

        labels = []
        encoded_features = []
        feature_names = []

        # Preparation phase
        data_prep_time = 0
        for samples, encoding in data:
            if not feature_names:
                feature_names = np.array(encoding.features)

            start = time.process_time_ns()
            if isinstance(samples, list):
                # Handle the list with multiple samples used together with
                # xarray DataArray encodings.
                for f in samples:
                    labels.append(f[PredictionField.GROUND_TRUTH])
                if len(encoded_features) == 0:
                    encoded_features = encoding
                else:
                    encoded_features = numpy.concatenate(
                        (encoded_features, encoding), axis=0
                    )
            else:
                labels.append(samples[PredictionField.GROUND_TRUTH])
                encoded_features.append(encoding[0])
            data_prep_time += time.process_time_ns() - start

        if len(encoded_features) == 0:
            log.warning("No data in encoded feature stream! Model is not trained")
            return

        # print(type(encoded_features[0]))
        # return

        start = time.process_time_ns()

        self.scaler.fit(encoded_features)
        scaled_encoded_features = self.scaler.transform(np.array(encoded_features))
        scaled_encoded_features = np.expand_dims(scaled_encoded_features, axis=2)
        labels = np.array(labels)

        self.model_instance = self.build_model()
        epochs = 50
        learning_rate = 0.0001
        batch_size = 1024
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.09, amsgrad=False)
        self.model_instance.compile(loss='binary_crossentropy', optimizer=optimizer,
                                    metrics=['accuracy'])  # here we specify the loss function
        self.model_instance.build(input_shape=(None, len(scaled_encoded_features[0]), 1))
        self.model_instance.summary()
        data_prep_time += time.process_time_ns() - start

        # Training phase
        training_start = time.process_time_ns()
        self.model_instance.fit(x=scaled_encoded_features, y=labels, epochs=epochs, batch_size=batch_size, verbose=2,
                                callbacks=[])
        training_time = time.process_time_ns() - training_start

        # self.encoded_features = np.array(scaled_encoded_features)
        self.train_data['X'] = np.array(encoded_features)
        self.train_data['scaled_X'] = scaled_encoded_features
        self.train_data['y'] = np.array(labels)
        self.train_data['feature_names'] = feature_names

        # Report
        report_performance(type(self).__name__ + "-preparation", log, len(labels),
                           data_prep_time)
        report_performance(type(self).__name__ + "-training", log, len(labels),
                           training_time)

        if not self.skip_saving_model:
            self.model_instance.save(f'{self.store_file}.h5', save_format='h5')
            dump(self.scaler, f'{self.store_file}-scaler')
            dump(self.model_instance, self.store_file)

    def load(self):
        self.model_instance = load_model(f'{self.store_file}.h5')
        self.scaler = load(f'{self.store_file}-scaler')

    def predict(self, data: EncodedSampleGenerator, **kwargs) -> SampleGenerator:
        sum_processing_time = 0
        sum_samples = 0
        test_data = []
        labels = []
        predicted_labels = []
        feature_names = []

        for sample, encoded_sample in data:
            if not feature_names:
                feature_names = np.array(encoded_sample.features)

            start_time_ref = time.process_time_ns()
            scaled_encoded_sample = self.scaler.transform(encoded_sample)
            scaled_encoded_sample = np.expand_dims(scaled_encoded_sample, axis=2)
            prediction = self.model_instance.predict(scaled_encoded_sample, batch_size=1024)

            if isinstance(sample, list):
                for i, s in enumerate(sample):
                    test_data.append(encoded_sample[i])
                    labels.append(s[PredictionField.GROUND_TRUTH])
                    predicted_labels.append(prediction[i][0] > 0.5)

                    s[PredictionField.MODEL_NAME] = self.model_name
                    s[PredictionField.OUTPUT_BINARY] = prediction[i][0] > 0.5
                    sum_processing_time += time.process_time_ns() - start_time_ref
                    sum_samples += 1
                    yield s
            else:
                test_data.append(encoded_sample)
                labels.append(sample[PredictionField.GROUND_TRUTH])
                predicted_labels.append(prediction[0][0] > 0.5)

                sample[PredictionField.MODEL_NAME] = self.model_name
                sample[PredictionField.OUTPUT_BINARY] = prediction[0][0] > 0.5
                sum_processing_time += time.process_time_ns() - start_time_ref
                sum_samples += 1
                yield sample

        # self.data_set = np.array(test_data)
        self.test_data['X'] = np.array(test_data)
        self.test_data['scaled_X'] = self.scaler.transform(test_data)
        self.test_data['y'] = np.array(labels)
        self.test_data['y_predict'] = np.array(predicted_labels)
        self.test_data['feature_names'] = feature_names
        self.feature_names = feature_names

        report_performance(type(self).__name__ + "-testing", log, sum_samples, sum_processing_time)

    def get_training_data(self):
        return self.train_data
        # return self.encoded_features

    def get_labels_name(self):
        return self.feature_names

    def get_test_data(self):
        return self.test_data
        # return self.data_set

    def predict_proba(self, data, **kwargs):
        data = np.transpose(data, axes=(0, 2, 1))
        predictions = self.model_instance.predict(data)
        return predictions

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        # dummy_input = np.random.random((1, *model.input_shape[1:]))  # Replace with your model's input shape
        # output = model(dummy_input)  # This will call the model and define the outputs
        print("output ", model.outputs[0])
        print("model.layers[-1].output ", model.get_layer("fc2").output)

        # model.summary()
        # print(model.inputs)

        print("img_array", img_array.shape)
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer('fc2').output]
            # [model.inputs], [model.get_layer(last_conv_layer_name).output, model.layers[-1].output]
        )
        # grad_model.summary()
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            tape.watch(last_conv_layer_output)
            tape.watch(preds)

            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            # print(pred_index[0])
            # print(preds)
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer

        print("class_channel", class_channel.shape, type(class_channel))
        print("last_conv_layer_output", last_conv_layer_output.shape, type(last_conv_layer_output))
        grads = tape.gradient(class_channel, last_conv_layer_output)
        print("grads ", grads)
        print("grads ", grads.shape)
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        # print ("pooled_grads", pooled_grads.shape)
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        # heatmap = tf.squeeze(heatmap)
        # print ("heatmap", heatmap.shape)
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def save_and_display_gradcam(self, feature_names, img, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        # img = keras.preprocessing.image.load_img(img_path)
        # img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        print("superimposed_img", superimposed_img.shape)
        temp2 = np.mean(superimposed_img, axis=-1)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        # print ("superimposed_img", superimposed_img.shape)
        # superimposed_img.save(cam_path)

        # Display Grad CAM
        # display(Image(cam_path))
        # plt.matshow(temp2)
        # plt.show()
        temp1 = np.squeeze(temp2)
        temp = np.mean(temp1, axis=0)
        return temp

    def explain_with_shap(self, train_data=None, test_data=None, number_of_samples=None, **kwargs):
        if train_data is None or test_data is None:
            log.error("Training data or Testing data is not provided")
            return

        if not number_of_samples:
            number_of_samples = len(test_data)
        elif number_of_samples <= 0:
            log.info(
                f"Number of samples for SHAP interpretation is set to {number_of_samples}, which is invalid. Therefore, numbe of samples for interpreration is re-assigned to number of samples in provided dataset")
            number_of_samples = len(test_data)

        background = train_data[:number_of_samples, :, :]
        explainer = shap.GradientExplainer(self.model_instance, background)
        shap_values = explainer.shap_values(test_data[:number_of_samples, :, :], nsamples=number_of_samples)
        shap_values = np.squeeze(shap_values)
        vals = np.abs(shap_values).mean(axis=0)
        return vals

    def explain_with_lime(self, train_data=None, test_data=None, number_of_samples=None, feature_names=None, **kwargs):
        if train_data is None or test_data is None:
            log.error("Training data or Testing data is not provided")
            return

        if not number_of_samples:
            number_of_samples = len(test_data)
        elif number_of_samples <= 0:
            log.info(
                f"Number of samples for SHAP interpretation is set to {number_of_samples}, which is invalid. Therefore, numbe of samples for interpreration is re-assigned to number of samples in provided dataset")
            number_of_samples = len(test_data)

        train_data = np.transpose(train_data, axes=(0, 2, 1))
        explainer = lime.lime_tabular.RecurrentTabularExplainer(train_data, mode='classification',
                                                                class_names=['0', '1'],
                                                                feature_names=feature_names, verbose=False)

        explain_values = np.empty((number_of_samples, len(feature_names)))
        for i in range(test_data[:number_of_samples, :].shape[0]):
            exp = explainer.explain_instance(test_data[i], self.predict_proba, num_features=len(feature_names),
                                             labels=(0,))
            exp_map = exp.as_map()
            feat = [exp_map[0][m][0] for m in range(len(exp_map[0]))]
            weight = [exp_map[0][m][1] for m in range(len(exp_map[0]))]
            mapping = dict(zip(feat, weight))
            sorted_dict = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[0])}
            explain_values[i] = list(sorted_dict.values())
        vals = np.abs(explain_values).mean(axis=0)
        return vals

    def explain_with_gradcam(self, train_data=None, test_data=None, number_of_samples=None):
        if test_data is None:
            log.error("Training data or Testing data is not provided")
            return

        print(test_data.shape)
        last_conv_layer_name = "conv0"
        img_array = test_data
        # Generate class activation heatmap
        heatmap = self.make_gradcam_heatmap(img_array, self.model_instance, last_conv_layer_name)
        vals = self.save_and_display_gradcam(self.feature_names, img_array, heatmap)
        return vals

    def explain_with_gbp(self, train_data=None, test_data=None, number_of_samples=None):
        if train_data is None:
            log.error("Training data or Testing data is not provided")
            return

        if not number_of_samples:
            number_of_samples = len(train_data)
        elif number_of_samples <= 0:
            log.info(
                f"Number of samples for SHAP interpretation is set to {number_of_samples}, which is invalid. Therefore, numbe of samples for interpreration is re-assigned to number of samples in provided dataset")
            number_of_samples = len(train_data)

        gb_model = Model(
            inputs=[self.model_instance.inputs],
            outputs=[self.model_instance.get_layer("fc1").output]
        )

        layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu

        with tf.GradientTape() as tape:
            inputs = tf.cast(train_data[:number_of_samples], tf.float32)
            tape.watch(inputs)
            outputs = gb_model(inputs)

        guided_grads = tape.gradient(outputs, inputs)[0]
        print(guided_grads)
        temp = np.squeeze(guided_grads)
        temp_abs = np.abs(temp)
        print("temp.shape", temp.shape)
        # mapping = dict(zip(self.feature_names, temp_abs))
        # sorted_dict = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[1])}
        vals = temp_abs
        return vals

    def predict_for_trustee(self, data):
        data = np.expand_dims(data, axis=2)
        raw_predictions = self.model_instance.predict(data, batch_size=1024)
        predictions = [raw_prediction[0] > 0.5 for raw_prediction in raw_predictions]
        return np.array(predictions)

    def get_model_for_trustee(self):
        return self

    def get_prediction_method_name_for_trustee(self):
        return "predict_for_trustee"

    def explain_with_trustee(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        class_names=None,
        feature_names=None,
        max_iter=5,
        num_pruning_iter=2,
        trustee_num_iter=50,
        trustee_num_stability_iter=20,
        trustee_sample_size=0.3,
        top_k=10,
        save_path=None,
        **kwargs
    ):
        if X_train is None or y_train is None:
            log.error("Training data is not provided")
            return

        # X_train = np.squeeze(X_train, axis=2)
        # X_test = np.squeeze(X_test, axis=2)

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        super().explain_with_trustee(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_names,
            max_iter,
            num_pruning_iter,
            trustee_num_iter,
            trustee_num_stability_iter,
            trustee_sample_size,
            top_k,
            save_path,
            **kwargs
        )

    def explain_with_mytrustee(
            self,
            X_train=None,
            y_train=None,
            X_test=None,
            y_test=None,
            class_names=None,
            feature_names=None,
            max_iter=5,
            num_pruning_iter=2,
            trustee_num_iter=50,
            trustee_num_stability_iter=20,
            trustee_sample_size=0.3,
            top_k=10,
            save_path=None,
            **kwargs
    ):
        if X_train is None or y_train is None:
            log.error("Training data is not provided")
            return

        # X_train = np.squeeze(X_train, axis=2)
        # X_test = np.squeeze(X_test, axis=2)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        super().explain_with_mytrustee(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_names,
            max_iter,
            num_pruning_iter,
            trustee_num_iter,
            trustee_num_stability_iter,
            trustee_sample_size,
            top_k,
            save_path,
            **kwargs
        )

    def explain_with_omnigradcam(self, train_data=None, test_data=None, number_of_samples=None):
        if test_data is None:
            log.error("Training data or Testing data is not provided")
            return

        print(test_data.shape)
        last_conv_layer_name = "conv0"
        explainer = GradCAM(
            model=self.model_instance,
            target_layer=self.model_instance.get_layer(last_conv_layer_name),
            preprocess_function=lambda x: x
        )

        explanations = explainer.explain(test_data)
        print(explanations)
