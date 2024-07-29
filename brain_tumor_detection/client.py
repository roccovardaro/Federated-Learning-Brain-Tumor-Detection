
import sys
import time

import flwr as fl
import dataset as dataset
import model as model
import os


class ClientFL(fl.client.NumPyClient):
    def __init__(self, train_data, test_data):
        self.model = model.create_model()  # Il modello è definito localmente
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        start_time=time.time()
        self.model.fit(self.train_data)
        end_time=time.time()
        print("tempo di addestramento", f"{end_time-start_time:.2f}")
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data)
        return loss, len(self.test_data), {"accuracy": accuracy, "loss": loss}


def main(dataset_dir):
    # si potrebbe passare il path del dataset diverso per ogni client in modo che ogni client gestisca un insieme diverso di campioni (VFL)
    img_height, img_width = 224, 224
    batch_size = 8
    train_set, test_set = dataset.load_data(dataset_dir=dataset_dir, img_height=img_height, img_width=img_width,
                                            batch_size=batch_size)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "NONE"

    # Inizializza il cliente
    client = ClientFL(train_set, test_set)

    # Avvia il processo federato
    fl.client.start_client(server_address="localhost:8080", client=client.to_client(),
                           grpc_max_message_length=536870912)


if __name__ == '__main__':
    data_path = sys.argv[1]
    main(data_path)
