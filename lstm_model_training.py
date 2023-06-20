import pandas as pd
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv("/content/gdrive/MyDrive/divided_data_by_classes/dataset.csv")
print(data.shape)

data_high = data[data['type'] == 'high'][:1000]
data_medium = data[data['type'] == 'medium'][:1000]
data_low = data[data['type'] == 'low'][:1000]
data_healthy = data[data['type'] == 'healthy'][:1000]
data = pd.concat([data_high, data_medium, data_low, data_healthy])
data.head()

def get_mfcc(path, target):
    x, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128)
    print(mfccs.shape)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title('MFCC of '+ target)
    plt.show()

def get_melspec(path, target, gender):
    x, sr = librosa.load(path)
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
    print(melspec.shape)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(melspec, sr=sr, x_axis='time')
    plt.title('Mel Spectrogram of '+ target)
    plt.show()

for i in np.unique(data['type']):
    sample = data[(data['type']==i)]
    get_mfcc(sample['path'].values[20], sample['type'].values[20])

def create_mfcc_features(path):
    mfccs = []
    x, sr = librosa.load(path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

from tqdm.notebook import tqdm

def safe_create_mfcc_features(x):
    try:
        return create_mfcc_features(x)
    except Exception as e:
        print(f"Error processing file {x}: {e}")
        return pd.Series()

tqdm.pandas()
X_df = pd.DataFrame(data['path'].progress_apply(lambda x: safe_create_mfcc_features(x)))
X_df = X_df.dropna()
X = X_df.copy()

X_df.loc[pd.isnull(X_df).any(1), :].index.values

X_df = X_df.dropna()
print(X_df.shape)
X_df.head()

X_df = pd.DataFrame(X_df['path'].to_list(), columns=[i for i in range(128)])
print(X_df)

encoder = LabelEncoder()
y = encoder.fit_transform(data['type'])
y.shape

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(np.array(X_df), y, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

y_train_encoded = to_categorical(y_train)
y_val_encoded = to_categorical(y_val)

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

from keras.optimizers import Adam

learning_rate = 0.001

opt = Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

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

print(x_train.shape)
history = model.fit(x_train, y_train_encoded, validation_data=(x_val, y_val_encoded), epochs=10, batch_size=32)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=y_pred_labels))

import pandas as pd
import seaborn as sn
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_labels)
df_cm = pd.DataFrame(cm, index=['healthy', 'low', 'medium', 'high'], columns=['healthy', 'low', 'medium', 'high'])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)

plt.show()