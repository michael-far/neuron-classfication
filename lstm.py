from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
import pandas as pd
import numpy as np
import os
import pickle

from sequence_gen import SequenceGen
from eda import calc_metrics, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
CLASSES = ['spiny', 'aspiny']
# CLASSES = ['2', '4', '5', '6']

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
    # model.add(Dense(7, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()

if __name__ == '__main__':
    # df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
    df = pd.read_pickle(db_file)
    df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
    df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
    try:
        df = df.drop('490278904_39') # Found to be bad samples
    except:
        pass
    if n_classes>2:
        df = df[df['layer'].isin(['2', '4', '5', '6'])]

    # train_dir = 'data/time_series/train/'
    # train_cells = [os.path.splitext(x)[0] for x in os.listdir(train_dir)]
    # train_idx = df.index.isin(train_cells)
    # test_dir = 'data/time_series/test/'
    # test_cells = [os.path.splitext(x)[0] for x in os.listdir(test_dir)]
    # test_idx = df.index.isin(test_cells)
    kf = StratifiedKFold(n_splits=5, random_state=42)
    stats = []
    for train_index, test_index in kf.split(df, df['dendrite_type']):
        train_generator = SequenceGen('data/time_series/', df.index[train_index], df['dendrite_type'][train_index].cat.codes)
        test_generator = SequenceGen('data/time_series/', df.index[test_index], df['dendrite_type'][test_index].cat.codes, batch_size=1,
                                     shuffle=False)
        # df['layer'] = df['layer'].apply(lambda x: to_categorical(x, num_classes=7))
        # train_generator = SequenceGen(train_dir, df.index[train_idx], df['layer'][train_idx])
        # test_generator = SequenceGen(test_dir, df.index[test_idx], df['layer'][test_idx])

        n_feats = 1
        n_steps = 450000
        model1 = get_LSTM_model(n_steps, n_feats)
        model1.fit_generator(generator=train_generator,
                             epochs=EPOCHS)
        model1.save('data/models/lstm')
        scaling_data = (train_generator.scale_mean, train_generator.scale_std)
        with open('data/models/lstm_scaling.p','wb') as f:
            pickle.dump(scaling_data, f)
        pred = model1.predict_generator(test_generator)
        pred = np.where(pred > 0.5, 1.0, 0.0)
        results = calc_metrics(df['dendrite_type'][test_index].cat.codes.values, pred)
        stats.append(results)
        reset_keras()
    mean_f1 = np.asarray([x[0] for x in stats]).mean()
    mean_accuracy = np.asarray([x[1] for x in stats]).mean()
    print('Accuracy: {}, f1: {}'.format(mean_accuracy, mean_f1))

    sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
    sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
    plot_confusion_matrix(sum_cm, 'LSTM model')
