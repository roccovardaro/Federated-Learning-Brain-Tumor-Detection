import datetime

import hydra
from omegaconf import DictConfig
import model as mod
import os
import flwr as fl
import history_transformer as ht
import customStrategy as customStr
import tensorflow as tf
import dataset as ds
import pandas as pd

accuracy = []
loss = []


def evaluate_metrics_aggregation_fn(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    losses = [m["loss"] for _, m in metrics]
    accuracy.append(sum(accuracies) / len(accuracies))
    loss.append(sum(losses) / len(losses))
    return {"accuracy": sum(accuracies) / len(accuracies), "loss": sum(losses) / len(losses)}


@hydra.main(config_path="conf", config_name="config_server", version_base=None)
def main(cfg: DictConfig):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "NONE"

    # 1. create model
    model = mod.create_modelCNN()

    # 2. define strategy

    stategy = customStr.CustomFedAvg(fraction_fit=cfg.fraction_fit, min_fit_clients=cfg.num_clients_per_round_fit,
                                     min_available_clients=cfg.num_clients,
                                     initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
                                     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                     num_rounds=cfg.num_rounds)

    # 3. configure Server
    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=stategy,
        grpc_max_message_length=536870912  # 512 MB
    )

    # 4. analysis of results

    transformer = ht.HistoryTransformer(history)
    df = transformer.to_dataframe()
    #df.to_csv("dataframe_history/history_FL_nc_"+str(cfg.num_clients_per_round_fit)+".csv", index=False)

    ht.plot_metrics(df)

    # 5. evaluate model from server
    x = datetime.datetime.now()
    date = str(x.day) + '_' + str(x.month) + '_' + str(x.year)
    model_after_FL = tf.keras.models.load_model("trained_models/model_final_"+date+".h5")

    test_set,_ = ds.load_data(dataset_dir='datasets/data_test_server', img_width=224, img_height=224, test_split=0, batch_size=32)
    loss_test_server, accuracy_test_server = model_after_FL.evaluate(test_set)
    print('loss', loss_test_server)
    print('accuracy_test_server', accuracy_test_server)

    print(accuracy)
    print(loss)


if __name__ == '__main__':
    main()
