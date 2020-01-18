import re
from typing import Union
import pandas as pd
import numpy as np
import os

import fastai.vision as vi
from helper_func import calc_metrics, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from model import Model

class CnnTransferModel(Model):
    def __init__(self, db: pd.DataFrame, num_layers: Union[int, None], epochs: int = 10, learn_rate: float = 0.01,
                 files_root: str = '', segment_length: float = 3.0):
        db = db[db['segment_length'] == segment_length]
        if num_layers == 0:
            num_layers = None
        if 'file_name' not in db.columns:
            db['file_name'] = [re.sub(f'_{str(segment_length)}', '', x) for x in db.index]

        super(CnnTransferModel, self).__init__(db, num_layers=num_layers,  epochs=epochs, files_root=files_root,
                                               segment_length=segment_length)

        self._learn_rate = learn_rate


# db_file = '/media/wd/data/cells/db.p'
# df = pd.read_pickle(db_file)
# df['layer'] = df['layer'].replace(['6a', '6b'], 6)
# df['layer'] = df['layer'].replace('2/3', 2)
# df['layer'] = df['layer'].astype('int')
# df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
#
# # CLASSES = ['2', '4', '5', '6']
# CLASSES = ['spiny', 'aspiny']
# n_classes = len(CLASSES)
# df['file_name'] = df.index
    def _create_model(self):
        """ fast.ai breaks this design"""
        return None

    def test(self):
        pass

    @staticmethod
    def pytprch_random_seed(seed_value):
        np.random.seed(seed_value)  # cpu vars
        vi.torch.manual_seed(seed_value)  # cpu  vars
        vi.torch.cuda.manual_seed(seed_value)
        vi.torch.cuda.manual_seed_all(seed_value)  # gpu vars
        vi.torch.backends.cudnn.deterministic = True  # needed
        vi.torch.backends.cudnn.benchmark = False

    def train_and_test(self, previous_accuracy: float = 0.0):
        kf = StratifiedKFold(n_splits=5, random_state=42)
        stats = []
        self.pytprch_random_seed(42)
        for train_index, test_index in kf.split(self._db, self._db['dendrite_type']):
            train = self._db.iloc[train_index]
            test = self._db.iloc[test_index]

            train = vi.ImageDataBunch.from_df(os.path.join(self._files_root, 'images', str(self._segment_length)),
                                              df=train, suffix='.png', valid_pct=0,
                                              fn_col='file_name', label_col='dendrite_type')
            train.normalize()
            test = vi.ImageDataBunch.from_df(os.path.join(self._files_root, 'images', str(self._segment_length)),
                                             df=test, suffix='.png', valid_pct=0,
                                             fn_col='file_name', label_col='dendrite_type')
            test.normalize()

            model = vi.cnn_learner(train, vi.models.resnet50, cut=self._num_layers, metrics=vi.accuracy)
            model.fit(epochs=self._epochs, lr=self._learn_rate)
            #
            # model.load('3dgadf_by_layers')
            pred = []
            for item in test.train_ds:
                    pred.append(str(model.predict(item[0])[0]))
            true = [str(x) for x in test.train_ds.y]
            results = calc_metrics(true, pred)
            stats.append(results)
        mean_f1 = np.asarray([x[0] for x in stats]).mean()
        mean_accuracy = np.asarray([x[1] for x in stats]).mean()
        print('Accuracy: {}, f1: {}'.format(mean_accuracy, mean_f1))
        if mean_accuracy > previous_accuracy:
            model.save('3dgadf_by_layers')
        sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
        sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
        params = {'cut_layers': self._num_layers, 'learn_rate': self._learn_rate, 'epochs': self._epochs}
        res = {'mean_accuracy': mean_accuracy, 'mean_f1': mean_f1}
        self._save_results(params, res, sum_cm, 'Transfer')
        return mean_accuracy
