from google.colab import drive

drive.mount('/content/gdrive')

import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np


def create_spectrogram(audio_file, save_dir, file_name):
    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name + '.png'), dpi=300, format='png')
    plt.clf()


def create_spectrograms_from_audio(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for class_dir in os.listdir(input_path):
        if not class_dir.endswith('.csv'):
            class_input_path = os.path.join(input_path, class_dir)
            class_output_path = os.path.join(output_path, class_dir)
            for audio_file in os.listdir(class_input_path):
                create_spectrogram(os.path.join(class_input_path, audio_file), class_output_path,
                                   audio_file.replace('.wav', ''))


create_spectrograms_from_audio('/content/gdrive/MyDrive/divided_data_by_classes/',
                               '/content/gdrive/MyDrive/spectograms/')

from tensorflow.keras.preprocessing import image
import tensorflow as tf


def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(
            tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))

    return images, labels


def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)


x = []
y = []

input_path = '/content/gdrive/MyDrive/divided_data_by_classes/'
output_path = '/content/gdrive/MyDrive/spectogram_by_classes/'

images, labels = load_images_from_path(output_path + 'healthy', 0)
show_images(images)

x += images[:1000]
y += labels[:1000]

images, labels = load_images_from_path(output_path + 'low', 1)
show_images(images)

x += images[:1000]
y += labels[:1000]

images, labels = load_images_from_path(output_path + 'medium', 2)
show_images(images)

x += images[:1000]
y += labels[:1000]

images, labels = load_images_from_path(output_path + 'high', 3)
show_images(images)

x += images[:1000]
y += labels[:1000]

import random

randomize = random.sample(range(len(x)), len(x))
x = [x[i] for i in randomize]
y = [y[i] for i in randomize]
print(len(x), len(y))

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


def rgb_to_gray(images):
    gray_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)
    return np.array(gray_images)


x_train_gray = rgb_to_gray(x_train)
x_test_gray = rgb_to_gray(x_test)
x_val_gray = rgb_to_gray(x_val)

x_train_norm = np.array(x_train_gray) / 255
x_test_norm = np.array(x_test_gray) / 255
x_val_norm = np.array(x_val_gray) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
y_val_encoded = to_categorical(y_val)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(x_train_norm.shape[1], x_train_norm.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
)

import visualkeras
from PIL import ImageFont

visualkeras.layered_view(model, legend=True)

from tensorflow.keras.callbacks import EarlyStopping

callback_ = EarlyStopping(monitor='val_loss', patience=3)
metrics = [tf.keras.metrics.Recall()]
adam1 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

history = model.fit(x_train_gray, y_train_encoded, validation_data=(x_val_gray, y_val_encoded), callbacks=[callback_],
                    epochs=10, batch_size=10)

model.evaluate(x_test_gray, y_test_encoded)

from numpy import argmax

y_prob = model.predict(x_test_gray)
y_classes = argmax(y_prob, axis=1)

import pandas as pd
import seaborn as sn

cm = confusion_matrix(np.argmax(y_test_encoded, axis=1), y_classes)
df_cm = pd.DataFrame(cm, index=['healthy', 'low', 'medium', 'high'], columns=['healthy', 'low', 'medium', 'high'])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(np.argmax(y_test_encoded, axis=1), y_classes,
                            target_names=['healthy', 'low', 'medium', 'high']))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)

plt.show()