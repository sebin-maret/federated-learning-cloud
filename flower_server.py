import os
from typing import List, Tuple, Union, Optional

import flwr as fl
import numpy as np
from flwr.common import Context, Metrics, ndarrays_to_parameters, FitRes, Parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy

import boto3

from settings import app_settings
from task import load_model

import tensorflow as tf


# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

            print(f"Pushing to S3 round {server_round} aggregated_ndarrays...")
            session = boto3.Session(aws_access_key_id=app_settings.aws_server_public_key,
                                    aws_secret_access_key=app_settings.aws_server_secret_key)
            s3 = session.resource('s3')
            classifier_bucket = s3.Bucket("saved-classifier-models")
            classifier_bucket.upload_file(f"round-{server_round}-weights.npz",f"round-{server_round}-weights.npz")

            if os.path.exists(f"round-{server_round}-weights.npz"):
                os.remove(f"round-{server_round}-weights.npz")

        return aggregated_parameters, aggregated_metrics

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Let's define the global model and pass it to the strategy
    # parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define the strategy
    strategy = strategy = SaveModelStrategy(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=None,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
