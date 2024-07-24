import tensorflow as tf
import matplotlib.pyplot as plt
img_height, img_width = 224, 224
batch_size = 8
dataset_dir = 'data'


def load_data():
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',  # Le etichette vengono inferite dalla struttura delle cartelle
        label_mode='binary',  # Per un problema binario (0 o 1)
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        class_names=['no', 'yes']  # Specifica l'ordine delle classi
    )
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_dataset


def split_dataset(dataset, val_split, test_split, dataset_size):
    """
    Divide il dataset in set di addestramento, validazione e test.
    """
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    # Dividi il dataset usando `take` e `skip`
    train_dataset = dataset.take(train_size // batch_size)
    val_dataset = dataset.skip(train_size // batch_size).take(val_size // batch_size)
    test_dataset = dataset.skip((train_size // batch_size) + (val_size // batch_size)).take(test_size // batch_size)

    return train_dataset, val_dataset, test_dataset
