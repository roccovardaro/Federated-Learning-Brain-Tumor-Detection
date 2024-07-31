import tensorflow as tf


#conv2d viene utilizzato per le convoluzioni
def create_modelCNN1():
    model = tf.keras.Sequential([
        #Aggiunge uno strato di input al modello. Questo strato accetta immagini con dimensioni 224x224 pixel
        # e 1 canali di colore (RGB)
        tf.keras.layers.Input(shape=(224, 224, 1)),  # Usa tf.keras.layers.Input
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        #ultimo layer della rete
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  #per la classificazione binaria
                  loss='binary_crossentropy',  # metrica utilizzata per correggere in modo ottimale i pesi della rete
                  metrics=['accuracy'])  # metrica che serve a me
    return model


#MODELLO CON RIDOTTO NUMERO DI PARAMETRI
def create_modelCNN2():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 1)),  # Usa tf.keras.layers.Input
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        # Unità singola con attivazione sigmoid per classificazione binaria
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_modelCNN():
    return create_modelCNN2()
