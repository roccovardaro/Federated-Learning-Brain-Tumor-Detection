import tensorflow as tf
from keras.src.layers import Rescaling
from matplotlib import pyplot as plt


def load_data(dataset_dir, img_height, img_width, batch_size, test_split=0.2, seed=42):
    # Carica l'intero dataset
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        color_mode="grayscale",
        shuffle=True, #le immagini vengono mescolate ad ogni epoca
        seed=seed
    )

    # Calcola il numero di batch per il training set e il test set
    total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
    test_size = int(test_split * total_batches)
    train_size = total_batches - test_size

    # Suddivide il dataset in training set e test set
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)

    # Normalizzazione delle immagini
    normalization_layer = Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, test_dataset


#METODO N0N UTILIZZATO
def partition_data(train_dataset, num_partitions, batch_size, val_ratio=0.1, seed=42):
    # Suddivide il training set in `num_partitions` partizioni
    train_partitions = []
    total_train_images = tf.data.experimental.cardinality(train_dataset).numpy()
    images_per_partition = total_train_images // num_partitions

    for i in range(num_partitions):
        start = i * images_per_partition
        end = start + images_per_partition if i < num_partitions - 1 else total_train_images
        partition = train_dataset.skip(start).take(end - start)
        train_partitions.append(partition)

    # Suddivisione train_dataset in train e valutation
    trainloaders = []
    valloaders = []

    normalization_layer = Rescaling(1.0 / 255)

    for partition in train_partitions:
        num_total = tf.data.experimental.cardinality(partition).numpy()
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        train_loader = partition.take(num_train).shuffle(10000, seed=seed).batch(batch_size).map(
            lambda x, y: (normalization_layer(x), y))
        val_loader = partition.skip(num_train).take(num_val).batch(batch_size).map(
            lambda x, y: (normalization_layer(x), y))

        trainloaders.append(train_loader)
        valloaders.append(val_loader)

    return trainloaders, valloaders


def getFirstImagesfromBatch(num_batch: int, dataset):
    # Il metodo take non consente l'accesso diretto, è necessario iterare
    for batch_images, batch_labels in dataset.take(num_batch):
        image_pixels = batch_images[0]
        label_pixels = batch_labels[0]
        # Il metodo .numpy() viene utilizzato per convertire un tensore di TensorFlow in un array NumPy
        plt.imshow(image_pixels.numpy().squeeze(), cmap='gray')
        plt.title(label_pixels.numpy().squeeze())
        plt.axis('off')
        plt.show()
    # Restituisce i pixel ognuno incapsulato in un array


if __name__ == '__main__':
    train_dataset, test_dataset = load_data(dataset_dir="datasets/data_test_server", img_height=224, img_width=224, batch_size=8,
                                            test_split=0.2)
    print(train_dataset)
    getFirstImagesfromBatch(num_batch=1, dataset=train_dataset)
