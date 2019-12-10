import pandas as pd
import numpy as np

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from model import Model
from helper_func import calc_metrics, plot_confusion_matrix


class FeatureLearner(Model):
    def __init__(self, db: pd.DataFrame, num_layers: int, num_nodes: int, batch_size: int = 64, epochs: int = 100,
                 files_root: str = ''):
        db = db.dropna(axis=1)
        irrelevant_columns = [c for c in db.columns if c.endswith('_i')] + \
                            [c for c in db.columns if c.endswith('index')] +\
                            ['layer', 'mean_clipped', 'structure_area_abbrev', 'sampling_rate']
        db = db.drop(irrelevant_columns, axis=1)

        super(FeatureLearner, self).__init__(db, num_layers=num_layers,
                                             num_nodes=num_nodes,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             files_root=files_root)


    def _create_model(self):
        n_feats = len(self._db.columns)-1  # minus 1 because we removed the label column
        model = Sequential()
        model.add(Dense(self._num_nodes, activation='relu', input_dim=n_feats))
        model.add(Dropout(0.5))
        for _ in range(self._num_layers):
            model.add(Dense(self._num_nodes, activation='relu'))
            model.add(Dropout(0.5))
        # model.add(Dense(y_train.shape[1],  activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def test(self):
        pass

    def train_and_test(self):
        df = self._db
        df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
        df['dendrite_type'] = df['dendrite_type'].cat.codes
        y = df.pop('dendrite_type')
        y = y.values.astype(float)
        x = df.values

        kf = StratifiedKFold(n_splits=5, random_state=42)
        stats = []
        for train_index, test_index in kf.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_index])
            y_train = y[train_index]
            x_test = scaler.transform(x[test_index])
            y_test = y[test_index]
            self.model.fit(x_train, y_train, epochs=self._epochs)
            self.model.save('data/models/ephys_dnn')
            pred = self.model.predict_classes(x_test)
            results = calc_metrics(y_test, pred)
            stats.append(results)
        mean_f1 = np.asarray([x[0] for x in stats]).mean()
        mean_accuracy = np.asarray([x[1] for x in stats]).mean()
        # print('Accuracy: {}, f1: {}'.format(mean_accuracy, mean_f1))

        sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
        sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
        params = {'num_layers': self._num_layers, 'num_nodes': self._num_nodes,
                  'batch_size': self._batch_size, 'epochs': self._epochs}
        res = {'mean_accuracy': mean_accuracy, 'mean_f1': mean_f1}
        self._save_results(params, res, sum_cm, 'DNN')
