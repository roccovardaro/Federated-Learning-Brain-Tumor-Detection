import flwr as fl
from typing import List, Tuple, Dict
import model as mod


def create_model_from_weights(weights: List):
    # Crea un modello dalle pesi aggregati (da implementare in base alla tua architettura di modello)
    model = mod.create_modelCNN()  # Definisci la tua architettura di modello
    model.set_weights(weights)
    return model


def save_model(parameters: fl.common.Parameters):
    # Converti i parametri in ndarray e imposta i pesi del modello
    weights = fl.common.parameters_to_ndarrays(parameters)
    model = create_model_from_weights(weights)
    model.save("model_final.h5")


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit: float, min_fit_clients: int, min_available_clients: int,
                 initial_parameters: fl.common.Parameters, evaluate_metrics_aggregation_fn, num_rounds: int):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_rounds = num_rounds  # Imposta il numero di round
        self.accuracy = []  # Lista per memorizzare l'accuratezza
        self.loss = []  # Lista per memorizzare la perdita

    def aggregate_fit(
            self, rnd: int, results: List[Tuple[fl.server.client_manager.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException]
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
        # Chiamaro il metodo aggregate_fit della classe base (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # all'ultimo round salvo il modello
        if rnd == self.num_rounds:
            save_model(aggregated_parameters)

        return aggregated_parameters, aggregated_metrics
