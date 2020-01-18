import os
import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

from neuron_classfier import NeuronClassifierSetup
from learning_lstm import LstmLearner
from learn_ephys_feats import FeatureLearner
from learning_fast_ai import CnnTransferModel
from learning_knn import Knn
from learning_naive_bayes import NaiveBayes
from learning_random_forest import RandomForest
from helper_func import random_search_helper

if __name__ == '__main__':
    species = 'mouse'
    df_location = f'/media/wd/data/{species}'
    segment_length = 3.0
    # segment_lengths = [ 3.0]

    if not os.path.isfile(df_location + '/db.p'):
        nc = NeuronClassifierSetup(species, df_location)
        # nc.mass_download()

        nc.create_data(segment_length)
    with open(df_location + '/db.p', 'rb') as f:
        df = pickle.load(f)

    # lstm_params = {'num_layers': [0, 1, 2], 'num_nodes': list(range(4, 61, 4)),
    #                'batch_size': list(range(4, 33, 8)), 'num_steps_in_time_series': list(range(300, 701, 50))}
    # mean_accuracy = 0
    # for res in random_search_helper(lstm_params, n_iters=10):
    #     print(res)
    #     lstm = LstmLearner(df, num_layers=res['num_layers'], num_nodes=res['num_nodes'], files_root='data/mouse',
    #                        num_steps_in_time_series=res['num_steps_in_time_series'], batch_size=res['batch_size'],
    #                        epochs=10)
    #     mean_accuracy = lstm.train_and_test(mean_accuracy)
    for i in range(5):
        mean_accuracy = 0
        cnn_params = {'num_layers': [0, 1, 2], 'learn_rate': list(np.arange(0.0001, 0.1, 0.005))}
        for res in cnn_params['learn_rate']:
            print(res)
            cnn = CnnTransferModel(df, epochs=100, num_layers=None, learn_rate=res,
                                   files_root='data/mouse')
            mean_accuracy = cnn.train_and_test(mean_accuracy)

    # dnn_params = {'num_layers': [1, 2, 3, 4], 'num_nodes': list(range(64, 513, 32)),
    #               'batch_size': list(range(4, 65, 8))}
    # mean_accuracy = 0
    # for res in random_search_helper(dnn_params, n_iters=50):
    #     feats = FeatureLearner(df, num_layers=res['num_layers'], num_nodes=res['num_nodes'],
    #                            files_root='data/mouse')
    #     feats.train_and_test()
    # feats = FeatureLearner(df, num_layers=1, num_nodes=416,
    #                            files_root='data/mouse')
    # mean_accuracy = feats.train_and_test(mean_accuracy)
    #
    # knn_params = {'k': [1, 2, 3, 4, 5]}
    # for k in knn_params['k']:
    #     knn = Knn(df, k, files_root='data/mouse')
    #     knn.train_and_test()
    #
    # rf_params = {'n_estimators': range(100, 1001, 50)}
    # for n in rf_params['n_estimators']:
    #     rf = RandomForest(df, n,  files_root='data/mouse')
    #     rf.train_and_test()
    #
    # nb = NaiveBayes(df,  files_root='data/mouse')
    # nb.train_and_test()

    # species = 'human'
    # df_location = 'data/human/'
    # if not os.path.isfile(df_location + '/db.p'):
    #     nc = NeuronClassifierSetup(species, df_location)
    #     # nc.mass_download()
    #     nc.create_data()
    # with open(df_location + '/db.p', 'rb') as f:
    #     df = pickle.load(f)
    # df = df.dropna(axis=1)
    # irrelevant_columns = [c for c in df.columns if c.endswith('_i')] + \
    #                      [c for c in df.columns if c.endswith('index')] + \
    #                      ['layer', 'mean_clipped', 'structure_area_abbrev', 'sampling_rate', 'mean_width']
    # df = df.drop(irrelevant_columns, axis=1)
    # df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
    # df['dendrite_type'] = df['dendrite_type'].cat.codes
    # y = df.pop('dendrite_type')
    # y = y.values.astype(float)
    # x = df.values
    # pred = feats.test(x)
    # from helper_func import calc_metrics, plot_confusion_matrix
    # import matplotlib.pyplot as plt
    # f1, acc, cm = calc_metrics(y, pred)
    # print(f1, acc)
    # sum_cm = cm / cm.astype(np.float).sum(axis=1).reshape(2, 1)
    # plot_confusion_matrix(sum_cm, 'Human Classfication')
    # plt.savefig('human.png')
    #


