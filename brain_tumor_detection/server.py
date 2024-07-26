import hydra
from omegaconf import DictConfig

import model as mod
import os
import flwr as fl
import result

accuracy = []
loss = []


def get_model_parameters():
    model = mod.create_model()
    return model.get_weights()


def evaluate_metrics_aggregation_fn(metrics):
    """Aggrega le metriche di valutazione dai client."""

    accuracies = [m["accuracy"] for _, m in metrics]
    losses = [m["loss"] for _, m in metrics]
    accuracy.append(sum(accuracies) / len(accuracies))
    loss.append(sum(losses) / len(losses))
    return {"accuracy": sum(accuracies) / len(accuracies), "loss": sum(losses) / len(losses)}


def strategy_selection(fraction_fit: float, min_fit_clients: int, min_available_clients: int):
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters()),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)


@hydra.main(config_path="conf", config_name="config_server", version_base=None)
def main(cfg: DictConfig):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "NONE"

    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)
    # Avvia il server
    fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=strategy_selection(cfg.fraction_fit, min_fit_clients=cfg.num_clients_per_round_fit,
                                    min_available_clients=cfg.num_clients),
        grpc_max_message_length=536870912  # 512 MB
    )

    print(accuracy)
    print(loss)
    result.plot_metrics(accuracy, loss, cfg.num_rounds)


if __name__ == '__main__':
    main()
