import os
import pickle
from numpy import linspace, arange

from neuron_classfier import NeuronClassifierSetup
from lstm import LstmLearner
from learn_ephys_feats import FeatureLearner
from learning_fast_ai import CnnTransferModel
from helper_func import random_search_helper

if __name__ == '__main__':
    species = 'mouse'
    df_location = '/media/wd/data/cells'
    if not os.path.isfile(df_location + '/db.p'):
        nc = NeuronClassifierSetup(species, df_location)
        nc.mass_download()
        nc.create_data()
    with open(df_location + '/db.p', 'rb') as f:
        df = pickle.load(f)

    lstm_params = {'num_layers': [0, 1, 2], 'num_nodes': list(range(4, 61, 4)),
                   'batch_size': list(range(4, 33, 8)), 'num_steps_in_time_series': list(range(300, 701, 50))}
    # for res in random_search_helper(lstm_params, n_iters=20):
    #     print(res)
    #     lstm = LstmLearner(df, num_layers=res['num_layers'], num_nodes=res['num_nodes'], files_root='data/mouse',
    #                        num_steps_in_time_series=res['num_steps_in_time_series'], batch_size=res['batch_size'],
    #                        epochs=10)
    #     lstm.train_and_test()

    # cnn_params = {'num_layers': [0, 1, 2], 'learn_rate': list(arange(0.0001, 0.1, 0.005))}
    # for res in random_search_helper(cnn_params, n_iters=50):
    #     print(res)
    #     cnn = CnnTransferModel(df, epochs=100, num_layers=None, learn_rate=res['learn_rate'],
    #                            files_root='data/mouse')
    #     cnn.train_and_test()

    dnn_params = {'num_layers': [1, 2, 3, 4], 'num_nodes': list(range(64, 513, 32)),
                  'batch_size': list(range(4, 65, 8))}
    for res in random_search_helper(dnn_params, n_iters=100):
        feats = FeatureLearner(df, num_layers=res['num_layers'], num_nodes=res['num_nodes'],
                               files_root='data/mouse')
        feats.train_and_test()

