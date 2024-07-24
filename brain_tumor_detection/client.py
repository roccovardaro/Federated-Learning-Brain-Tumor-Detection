import flwr as fl
import dataset as dataset
import model as model
import os

class ClientFL(fl.client.NumPyClient):
    def __init__(self, train_data, val_data, test_data):
        self.model = model.create_model()  # Il modello è definito localmente
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data, validation_data=self.val_data, epochs=1)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data)
        return loss, len(self.test_data), {"accuracy": accuracy}


data = dataset.load_data()
train_dataset, val_dataset, test_dataset = dataset.split_dataset(dataset=data, val_split=0.05, test_split=0.05)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "NONE"

# Inizializza il cliente
client = ClientFL(train_dataset, val_dataset, test_dataset)

# Avvia il processo federato
fl.client.start_client(server_address="localhost:8080", client=client.to_client(),grpc_max_message_length=536870912)
