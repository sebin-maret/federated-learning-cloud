"""tfexample: A Flower / TensorFlow app."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tensorflow.keras.callbacks import ModelCheckpoint
from  tensorflow.keras.callbacks import EarlyStopping
import numpy as np

from task import load_data, load_model

import tensorflow as tf

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
    ):
        self.model = load_model(learning_rate)
        self.train_generator, self.test_generator = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config):
        """
        get the model parameters and return them as a list of NumPy ndarray’s
        (which is what flwr.client.NumPyClient expects)

        args:
            config: config (dict)

        return:
            parameters: parameters (list)
        """
        weights = self.model.get_weights()
        return weights

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        checkpoint = ModelCheckpoint('vgg16_best_weights.h5',
                                     monitor='val_accuracy',
                                     #  monitor='val_f1_score',
                                     verbose=1,
                                     mode='max',
                                     save_best_only=True)

        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              restore_best_weights=True,
                              patience=5)
        callbacks_list = [checkpoint, early]
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.test_generator,
            callbacks=[callbacks_list],
            verbose=self.verbose,
            shuffle=True)

        return self.model.get_weights(), len(self.train_generator.labels), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        test_result = self.model.evaluate(self.test_generator)
        loss, accuracy=np.round(test_result[0], 4), np.round(test_result[1], 3)
        return loss, len(self.test_generator.labels), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    data = load_data(batch_size)



    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)