from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from helper_func import plot_confusion_matrix


class Model(ABC):
    def __init__(self, db: pd.DataFrame, num_layers: int = 2, num_nodes: int = 256, batch_size: int = 64,
                 epochs: int = 10, files_root: str = ''):
        self._db = db
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._batch_size = batch_size
        self._epochs = epochs
        self.model = self._create_model()
        self._files_root = files_root

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def train_and_test(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def _save_results(self, params: dict, results: dict, sum_cm: np.ndarray, task: str):
        headers = [str(x) for x in params.keys()] + [str(x) for x in results.keys()]
        file_prefix = os.path.join(self._files_root, 'results', task)
        param_prefix = '_'.join([str(x) for x in params.values()])
        results_full_file_name = file_prefix + '_results.csv'
        if not os.path.isfile(results_full_file_name):
            write_headers = True
        else:
            write_headers = False
        with open(results_full_file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_headers:
                writer.writeheader()
            writer.writerow({**params, **results})
        plot_confusion_matrix(sum_cm, task + ' model')
        plt.savefig('{}_cm_{}.png'.format(file_prefix, param_prefix))