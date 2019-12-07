import os
import pickle
from neuron_classfier import NeuronClassifier
from lstm import LstmLearner
from learn_ephys_feats import FeatureLearner
from learning_fast_ai import CnnTransferModel
from sklearn.model_selection import RandomizedSearchCV

from helper_func import random_search_helper

if __name__ == '__main__':
    species = 'mouse'
    df_location = '/media/wd/data/cells'
    if not os.path.isfile(df_location + '/db.p'):
        nc = NeuronClassifier(species, df_location)
        nc.mass_download()
        nc.create_data()
    with open(df_location + '/db.p', 'rb') as f:
        df = pickle.load(f)

    lstm_params = {'num_layers':[0, 1, 2], 'num_nodes': list(range(4, 21, 2)),
                   'batch_size': list(range(4, 33, 8)), 'num_steps_in_time_series': list(range(300, 701, 50))}
    for res in random_search_helper(lstm_params):
        print(res)
        lstm = LstmLearner(df, num_layers=res['num_layers'], num_nodes=res['num_nodes'], files_root='data/mouse',
                           num_steps_in_time_series=res['num_steps_in_time_series'], batch_size=res['batch_size'],
                           epochs=2)
        lstm.train_and_test()

    # cnn = CnnTransferModel(df,epochs=100, num_layers=None, learn_rate=0.001, files_root='data/mouse')
    # cnn.train_and_test()

    # feats = FeatureLearner(df, 2, 256, files_root='data/mouse')
    # feats.train_and_test()

