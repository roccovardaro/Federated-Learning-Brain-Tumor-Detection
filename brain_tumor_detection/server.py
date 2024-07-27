import hydra
from matplotlib import pyplot as plt, ticker
from omegaconf import DictConfig
import model as mod
import os
import flwr as fl
import history_transformer as ht

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

@hydra.main(config_path="conf", config_name="config_server", version_base=None)
def main(cfg: DictConfig):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "NONE"


    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.fraction_fit,
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_available_clients=cfg.num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters()),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)

    # Avvia il server
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=strategy,
        grpc_max_message_length=536870912  # 512 MB
    )

    #analysis of results

    transformer = ht.HistoryTransformer(history)
    df = transformer.to_dataframe()
    # Creare il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['accuracy'], label='Accuracy')
    plt.plot(df['epoch'], df['loss'], label='Loss')

    # Impostare l'asse X per mostrare solo numeri interi
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Etichette e titolo
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training Metrics per Round')
    plt.legend()
    plt.grid(True)

    # Visualizzare il grafico
    plt.show()

if __name__ == '__main__':
    main()
