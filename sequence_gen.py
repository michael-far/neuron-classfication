from keras.utils import Sequence
import numpy as np
from pyts.approximation.paa import PiecewiseAggregateApproximation
from scipy.signal import find_peaks
import os


N_FEATS = 2


class SequenceGen(Sequence):
    scale_mean = None
    scale_std = None
    'Generates data for Keras'
    def __init__(self, data_dir, ids, labels, batch_size=16, dim=450,
                 shuffle=True):
        'Initialization'
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_classes = len(np.unique(labels))
        self.shuffle = shuffle
        if self.scale_mean is None:
            SequenceGen.scale_mean = np.zeros(N_FEATS)
            SequenceGen.scale_std = np.zeros(N_FEATS)
        self.epoch = 0
        self.index_complete = None
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.index_complete += 1
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        ids_temp = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(ids_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.epoch += 1
        self.index_complete = 0

    def __data_generation(self, ids_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, N_FEATS, self.dim))
        y = np.empty((self.batch_size, 1))
        paa = PiecewiseAggregateApproximation(window_size=None, output_size=self.dim)
        # Generate data
        for i, ID in enumerate(ids_temp):
            # Store sample
            sample = np.load(self.data_dir + ID + '.npy')
            # X[i, 0] = paa.fit_transform(sample[0:150000].reshape((1, -1))).reshape((1, -1))
            # X[i, 0] = paa.fit_transform(sample[150000:300000].reshape((1, -1))).reshape((1, -1))
            # X[i, 1] = paa.fit_transform(sample[300000:450000].reshape((1, -1))).reshape((1, -1))
            X[i, 0] = self._peak_quantize(sample[150000:300000])
            X[i, 1] = self._peak_quantize(sample[300000:450000])


            # Store class
            y[i] = self.labels[ID]

        if self.batch_size > 1:
            scaled_x = self.scale_batch(X)
        else:
            scaled_x = self.scale(X)
        X = scaled_x.reshape((X.shape[0], self.dim, N_FEATS))
        # X = X.reshape((X.shape[0], self.dim, N_FEATS))
        return X, y

    def scale_batch(self, x):
        mean = np.mean(x, axis=(0, 2))
        std = np.std(x, axis=(0, 2))
        samples_seen = (self.epoch-1) * len(self.indexes) + (self.index_complete-1) * self.batch_size
        # https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        SequenceGen.scale_std = np.sqrt(samples_seen/(samples_seen + self.batch_size)* SequenceGen.scale_std**2 +
                                        self.batch_size/(samples_seen + self.batch_size)*std**2 +\
                                        samples_seen * self.batch_size/(samples_seen + self.batch_size)**2 *
                                        (SequenceGen.scale_mean - mean)**2)
        SequenceGen.scale_mean = ((samples_seen * SequenceGen.scale_mean) + (self.batch_size * mean)) / (samples_seen + self.batch_size)
        return self.scale(x)

    def scale(self, x):
        x = np.subtract(x, self.scale_mean.reshape(N_FEATS, 1))
        x = np.divide(x, self.scale_std.reshape(N_FEATS, 1))
        return x

    def _peak_quantize(self, signal: np.ndarray):
        signal_parts = np.array_split(signal, self.dim)
        return np.array([float(len(find_peaks(x)[0])) for x in signal_parts])


if __name__ == '__main__':
    import pandas as pd

    db_file = '/media/wd/data/cells/db.p'
    data_dir = 'data/time_series/train/'
    # df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
    df = pd.read_pickle(db_file)
    df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
    df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
    cells = [os.path.splitext(x)[0] for x in os.listdir(data_dir)]
    s = SequenceGen(data_dir, df.index[df.index.isin(cells)], df['dendrite_type'].cat.codes)
    a = s.__getitem__(1)
    print(a)
    a = s.__getitem__(2)
    print(a)