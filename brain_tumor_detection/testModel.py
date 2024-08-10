import numpy as np
import tensorflow as tf
from PIL import Image
import dataset as ds


def preprocess_image(image_path, target_size):
    # Carica l'immagine
    img = Image.open(image_path)
    # Converti in scala di grigi
    img = img.convert('L')
    # Ridimensiona l'immagine
    img = img.resize(target_size)
    # Converti l'immagine in un array numpy
    img_array = np.array(img)
    # Normalizza i valori dei pixel (da 0-255 a 0-1)
    img_array = img_array / 255.0
    # Aggiungi una dimensione batch e una dimensione del canale
    img_array = np.expand_dims(img_array, axis=0)  # Aggiungi dimensione batch
    img_array = np.expand_dims(img_array, axis=-1)  # Aggiungi dimensione canale
    return img_array


def value_prediction(predictions):
    if predictions[0][0] > 0.5:
        print('Healthy', predictions[0][0])
    else:
        print('Brain Tumor', predictions[0][0])
    return


def predict_image(image_path, name_model):
    model_after_FL = tf.keras.models.load_model(name_model)
    target_size = (224, 224)  # Adatta alla dimensione che il tuo modello richiede
    preprocessed_image = preprocess_image(image_path, target_size)
    predictions = model_after_FL.predict(preprocessed_image)
    print(value_prediction(predictions))


def main():
    model_name = "trained_models/model_final.h5"
    dataset_path = 'datasets/data_test_server'

    model = tf.keras.models.load_model(model_name)
    test_set, _ = ds.load_data(dataset_path, 224, 224, test_split=0, batch_size=32)
    model.evaluate(test_set)

    #predict single image
    #imagePath = 'datasets/brain_tumor_dataset2/Healthy/no8.jpg'
    #predict_image(image_path=imagePath, name_model=model_name)


if __name__ == '__main__':
    main()