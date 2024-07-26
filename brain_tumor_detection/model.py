import tensorflow as tf



def create_model():
    model = tf.keras.Sequential([
        #Aggiunge uno strato di input al modello. Questo strato accetta immagini con dimensioni 224x224 pixel
        # e 1 canali di colore (RGB)
        tf.keras.layers.Input(shape=(224, 224, 3)),  # Usa tf.keras.layers.Input
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

