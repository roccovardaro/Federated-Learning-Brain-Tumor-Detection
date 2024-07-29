import pandas as pd
from flwr.server.history import History


class HistoryTransformer:
    def __init__(self, history: History):
        self.history = history

    def to_dataframe(self) -> pd.DataFrame:
        # Estrazione dei dati di loss
        loss_data = [(rnd, loss) for rnd, loss in enumerate(self.history.losses_distributed, start=1)]

        # Estrazione dei dati di metrics
        metrics = self.history.metrics_distributed
        accuracy_data = [(rnd, acc) for rnd, acc in metrics["accuracy"]]
        loss_metric_data = [(rnd, loss) for rnd, loss in metrics["loss"]]

        # Combinare i dati in un unico dizionario
        combined_data = {
            'epoch': [x[0] for x in accuracy_data],
            'accuracy': [x[1] for x in accuracy_data],
            'loss': [x[1] for x in loss_metric_data]
        }

        # Creare un DataFrame combinato
        df_combined = pd.DataFrame(combined_data)

        return df_combined
