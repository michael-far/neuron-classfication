import re
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

from sequence_gen import SequenceGen
from helper_func import calc_metrics, plot_confusion_matrix
from model import Model


class LstmLearner(Model):
    def __init__(self, db: pd.DataFrame, num_layers: int, num_nodes: int, batch_size: int = 4, epochs: int = 10,
                 num_steps_in_time_series: int = 450000, files_root: str = '', segment_length: float = 3.0):
        self._n_steps = num_steps_in_time_series
        db = db[db['segment_length'] == segment_length]
        db.index = [re.sub(f'_{str(segment_length)}', '', x) for x in db.index]
        db['file_name'] = db.index
        super(LstmLearner, self).__init__(db, num_layers=num_layers,
                                          num_nodes=num_nodes,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          files_root=files_root, segment_length=segment_length)

    def train(self):
        pass

    def test(self):
        pass

    def _create_model(self):
        n_feats = 2
        model = Sequential()
        # model.add(LSTM(hidden_layer_size, input_shape=(n_steps, n_features), return_sequences=True))
        if self._num_layers > 0:
            for _ in range(self._num_layers):
                model.add(LSTM(self._num_nodes, activation='relu', return_sequences=True,
                              input_shape=( self._n_steps, n_feats)))
                model.add(Dropout(0.2))
        model.add(LSTM(self._num_nodes, activation='relu', input_shape=(self._n_steps, n_feats)))
        # model.add(Dropout(0.2))
        # model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.add(Dense(7, activation='softmax'))
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def __reset_keras():
        sess = get_session()
        clear_session()
        sess.close()

    def train_and_test(self, previous_accuracy: float = 0.0):
        # df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
        # df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        self._db['dendrite_type'] = pd.Categorical( self._db['dendrite_type'])
        try:
            self._db = self._db.drop('490278904_39') # Found to be bad sample
        except:
            pass
        kf = StratifiedKFold(n_splits=5, random_state=42)
        stats = []
        for train_index, test_index in kf.split( self._db,  self._db['dendrite_type']):
            train_generator = SequenceGen(os.path.join(self._files_root, 'time_series', str(self._segment_length)),
                                          self._db.index[train_index], self._db['dendrite_type'][train_index].cat.codes,
                                          batch_size=self._batch_size, dim=self._n_steps)
            test_generator = SequenceGen(os.path.join(self._files_root, 'time_series' , str(self._segment_length)),
                                         self._db.index[test_index], self._db['dendrite_type'][test_index].cat.codes,
                                         dim=self._n_steps , batch_size=1, shuffle=False)
            self.model.fit_generator(generator=train_generator, epochs=self._epochs)
            self.model.reset_states()
            # from keras.models import load_model
            # self.model = load_model('data/mouse/models/lstm.model')
            # with open(os.path.join(self._files_root, 'models', 'lstm_scaling.p'), 'rb') as f:
            #     SequenceGen.scale_mean, SequenceGen.scale_std = pickle.load(f)
            pred = self.model.predict_generator(test_generator)
            pred = np.where(pred > 0.5, 1.0, 0.0)
            self.model.reset_states()
            results = calc_metrics(self._db['dendrite_type'][test_index].cat.codes.values, pred)
            stats.append(results)
            # self.__reset_keras()
        mean_f1 = np.asarray([x[0] for x in stats]).mean()
        mean_accuracy = np.asarray([x[1] for x in stats]).mean()
        if mean_accuracy > previous_accuracy:
            self.model.save(os.path.join(self._files_root, 'models', 'lstm.model'))
            scaling_data = (train_generator.scale_mean, train_generator.scale_std)
            with open(os.path.join(self._files_root, 'models', 'lstm_scaling.p'), 'wb') as f:
                pickle.dump(scaling_data, f)
        sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
        sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
        params = {'num_layers': self._num_layers, 'num_nodes': self._num_nodes,
                  'batch_size': self._batch_size, 'epochs': self._epochs,
                  'n_steps': self._n_steps}
        res = {'mean_accuracy': mean_accuracy, 'mean_f1': mean_f1}
        self._save_results(params, res, sum_cm, 'LSTM')
        return mean_accuracy
