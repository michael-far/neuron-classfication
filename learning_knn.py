import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from model import Model
from helper_func import calc_metrics, plot_confusion_matrix


class Knn(Model):
    def __init__(self, db: pd.DataFrame, k: int, files_root: str = '', segment_length: float = 3.0):
        db = db[db['segment_length'] == segment_length]
        db = db.dropna(axis=1)
        irrelevant_columns = [c for c in db.columns if c.endswith('_i')] + \
                             [c for c in db.columns if c.endswith('index')] + \
                             ['layer', 'mean_clipped', 'structure_area_abbrev', 'sampling_rate', 'segment_length']
        db = db.drop([x for x in irrelevant_columns if x in db.columns], axis=1)
        self._k = k
        super(Knn, self).__init__(db, files_root=files_root, segment_length=segment_length)

    def _create_model(self):
        model = KNeighborsClassifier(self._k)
        return model

    def test(self):
        pass

    def train_and_test(self, previous_accuracy: float = 0.0):
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
            self.model.fit(x_train, y_train)
            pred = self.model.predict(x_test)
            results = calc_metrics(y_test, pred)
            stats.append(results)
        mean_f1 = np.asarray([x[0] for x in stats]).mean()
        mean_accuracy = np.asarray([x[1] for x in stats]).mean()
        # print('Accuracy: {}, f1: {}'.format(mean_accuracy, mean_f1))

        sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
        sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
        params = {'k': self._k}
        res = {'mean_accuracy': mean_accuracy, 'mean_f1': mean_f1}
        self._save_results(params, res, sum_cm, 'Knn')
