import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import shutil
import random

db_file = '/media/wd/data/cells/db.p'
# df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
df = pd.read_pickle(db_file)
df = df.astype('str')

def get_common_sample_frequency(df: pd.DataFrame) -> float:
    df['sampling_rate'] = df['sampling_rate'].astype('float')
    return df['sampling_rate'].mode()

def test_train_split(class_folder: str, test_percent=0.2):
    pwd = '/'.join(class_folder.split('/')[:-1])
    class_name = class_folder.split('/')[-1]
    test_percent = 1 - test_percent
    for folder in ['/train', '/test']:
        if not os.path.exists(pwd + folder):
            os.mkdir(pwd + folder)
        os.mkdir('{}{}/{}'.format(pwd, folder, class_name))
        files = [f for f in os.listdir(class_folder)]
        files_to_transfer = random.sample(files, int(len(files)*test_percent))
        for file in files_to_transfer:
            os.rename('{}/{}'.format(class_folder, file), '{}{}/{}/{}.png'.format(pwd, folder, class_name, file))
        test_percent = 1



if __name__ == '__main__':
    test_train_split('/home/michael/PycharmProjects/neuron-classfication/data/images/3dgaf_copy/aspiny', test_percent = 0.2)
    # print('Missing value count: {}'.format(df.isnull().sum().sum()))
    #
    # df['layer'] = df['layer'].replace(['6a', '6b'], 6)
    # df['layer'] = df['layer'].replace('2/3', 2)
    # df['layer'] = df['layer'].astype('float')
    #
    # df['layer'].hist(grid=False)
    # plt.title('Layers')
    # plt.show()
    #
    #
    # df['structure_area_abbrev'].value_counts().plot(kind='bar')
    # plt.title('Brain Regions')
    # plt.show()
    #
    # df['sampling_rate'] = df['sampling_rate'].astype('float')
    # df['sampling_rate'].hist(grid=False)
    # plt.title('sampling Rate')
    # plt.show()
    #
    #
    # df['dendrite_type'].value_counts().plot(kind='bar')
    # plt.title('Dendrite type')
    # plt.show()
    #
