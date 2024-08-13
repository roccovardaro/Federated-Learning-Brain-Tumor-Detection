import sys

import flwr as fl

import dataset as dataset
import model as model
import os


class ClientFL(fl.client.NumPyClient):
    def __init__(self, train_data, val_data, test_data, client_id):
        self.model = model.create_modelCNN()  # Il modello è definito localmente
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.id_client = client_id
        self.count_round = 0

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.train_data, validation_data=self.val_data)

        self.count_round += 1
        #history_df = pd.DataFrame(history.history)
        #history_df.to_csv('history_fit_id_' + str(self.id_client) + '_round_' + str(self.count_round) + '.csv',index=False)

        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data)
        return loss, len(self.test_data), {"accuracy": accuracy, "loss": loss}


def main(dataset_dir, id_client):
    img_height, img_width = 224, 224
    batch_size = 64

    train_set, val_set, test_set = dataset.load_data_with_validation(dataset_dir=dataset_dir, img_height=img_height,
                                                                     img_width=img_width, batch_size=batch_size,
                                                                     val_split=0.1, test_split=0.2)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "NONE"

    # Inizializza il cliente
    client = ClientFL(train_set, val_set, test_set, id_client)


    # Avvia il processo federato
    fl.client.start_client(server_address="localhost:8080", client=client.to_client(),
                           grpc_max_message_length=536870912)


if __name__ == '__main__':
    data_path = sys.argv[1]
    id_client = sys.argv[2]
    main(data_path, id_client)
