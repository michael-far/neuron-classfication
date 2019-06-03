from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle
from sequence_gen import SequenceGen
import os

db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
CLASSES = ['spiny', 'aspiny']
n_classes = len(CLASSES)
EPOCHS = 10

def get_LSTM_model(n_features, n_steps,  hidden_layer_size=64):
    model = Sequential()
    # model.add(LSTM(hidden_layer_size, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(LSTM(hidden_layer_size, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
    df = pd.read_pickle(db_file)
    df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
    df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
    try:
        df = df.drop('490278904_39') # Found to be bad samples
    except:
        pass

    train_dir = 'data/time_series/train/'
    train_cells = [os.path.splitext(x)[0] for x in os.listdir(train_dir)]
    train_idx = df.index.isin(train_cells)
    test_dir = 'data/time_series/test/'
    test_cells = [os.path.splitext(x)[0] for x in os.listdir(test_dir)]
    test_idx = df.index.isin(test_cells)

    train_generator = SequenceGen(train_dir, df.index[train_idx], df['dendrite_type'][train_idx].cat.codes)
    test_generator = SequenceGen(test_dir, df.index[test_idx], df['dendrite_type'][test_idx].cat.codes)

    n_feats = 1
    n_steps = 450000
    model1 = get_LSTM_model(n_steps, n_feats)
    model1.fit_generator(generator=train_generator,
                         validation_data=test_generator, epochs=EPOCHS)
    model1.save('data/models/lstm')
