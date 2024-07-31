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
        print('Healthy')
    else:
        print('Brain Tumor')
    return


def predict_image(image_path):
    model_after_FL = tf.keras.models.load_model("model_final_98.h5")
    target_size = (224, 224)  # Adatta alla dimensione che il tuo modello richiede
    preprocessed_image = preprocess_image(image_path, target_size)
    predictions = model_after_FL.predict(preprocessed_image)
    print(value_prediction(predictions))


model_after_FL = tf.keras.models.load_model("model_final_98.h5")

#train_set, test_set = ds.load_data('data', 224, 224, 32)
#model_after_FL.evaluate(train_set)


predict_image('data_example/foto2.png')
