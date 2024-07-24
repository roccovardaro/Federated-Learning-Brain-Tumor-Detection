import model as mod
import os
import flwr as fl


def get_model_parameters():
    model = mod.create_model()
    return model.get_weights()


def evaluate_metrics_aggregation_fn(metrics):
    """Aggrega le metriche di valutazione dai client."""
    accuracies = [m["accuracy"] for _, m in metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}


# Definisci la strategia del server
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters()),
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn

)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "NONE"

# Definisci la configurazione del server
server_config = fl.server.ServerConfig(num_rounds=2)  # Ferma il server dopo 1 round

# Avvia il server
fl.server.start_server(
    server_address="localhost:8080",
    config=server_config,
    strategy=strategy,
    grpc_max_message_length=536870912  # 512 MB
)