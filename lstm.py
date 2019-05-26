from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle


db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
CLASSES = ['spiny', 'aspiny']
n_classes = len(CLASSES)
EPOCHS = 10

def train_LSTM_model(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[2]
    n_steps = x_train.shape[1]
    hidden_layer_size = 32
    model = Sequential()
    model.add(LSTM(hidden_layer_size, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_layer_size*2, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=EPOCHS)
    return model



if __name__ == '__main__':
    train_datagen = ImageDataGenerator(validation_split=0.2)  # included in our dependencies

    train_generator = train_datagen.flow_from_directory('data/images/3dgaf',
                                                        color_mode='rgb',
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        shuffle=True,
                                                        subset='training')  # )

    validation_generator = train_datagen.flow_from_directory('data/images/3dgaf',
                                                             target_size=(224, 224),
                                                             color_mode='rgb',
                                                             batch_size=BATCH_SIZE,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                             subset='validation')  # set as validation data