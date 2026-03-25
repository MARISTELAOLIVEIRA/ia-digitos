import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# carregar MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizar
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape para CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 🔥 carregar feedback se existir
if os.path.exists("feedback"):
    X_new = []
    y_new = []

    for file in os.listdir("feedback"):
        data = np.load(f"feedback/{file}")
        label = int(file.split("_")[1])

        X_new.append(data)
        y_new.append(label)

    if len(X_new) > 0:
        X_new = np.array(X_new)
        y_new = np.array(y_new)

        x_train = np.concatenate((x_train, X_new))
        y_train = np.concatenate((y_train, y_new))

        print(f"📊 Adicionados {len(X_new)} novos exemplos!")

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(x_train)

# modelo CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# treinar
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10)

# avaliar
loss, acc = model.evaluate(x_test, y_test)
print("Acurácia:", acc)

# salvar
model.save("model.keras")
print("✅ Modelo atualizado!")