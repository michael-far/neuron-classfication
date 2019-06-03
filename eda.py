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
# df = df.astype('str')
# CLASSES = ['2', '4', '5', '6']
CLASSES = ['spiny', 'aspiny']
n_classes = len(CLASSES)


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
            os.rename('{}/{}'.format(class_folder, file), '{}{}/{}/{}'.format(pwd, folder, class_name, file))
        test_percent = 1


def move_to_class_folder(cell, sort_by, out_dir, file_type='png'):
    class_name = df.loc[cell][sort_by]
    class_dir = out_dir + str(class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    os.rename('{}{}.{}'.format(out_dir, cell, file_type), '{}{}/{}.{}'.format(out_dir, class_name, cell, file_type))


def arrange_files(out_dir, ext):
    cells = df.index[df['dendrite_type'].isin(['spiny', 'aspiny'])]
    for cell in cells:
        # convert_cell(cell, df, out_dir)
        move_to_class_folder(cell, 'dendrite_type', out_dir, ext)
    for clas in [out_dir + x for x in CLASSES]:
        test_train_split(clas)    # print('Missing value count: {}'.format(df.isnull().sum().sum()))

def pca_plot(x, y):
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    pca = PCA(n_components=3)
    pca.fit(x)
    X_r = pca.transform(x)
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for color, i in zip(['navy', 'turquoise'], [0, 1]):
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2], color=color, alpha=0.8, lw=2)
    plt.show()

if __name__ == '__main__':
    out_dir = 'data/time_series/'
    arrange_files(out_dir, ext='npy')
    #
    # df['layer'] = df['layer'].replace(['6a', '6b'], 6)
    # df['layer'] = df['layer'].replace('2/3', 2)
    # df['layer'] = df['layer'].astype('float')
    #
    # df['layer'].hist(grid=False)
    # plt.title('Layers')
    # plt.show()
    # plt.savefig('layers.png')
    #
    # df['structure_area_abbrev'].value_counts().plot(kind='bar')
    # plt.title('Brain Regions')
    # plt.show()
    # plt.savefig('regions.png')
    # plt.show()
    #
    # df['sampling_rate'] = df['sampling_rate'].astype('float')
    # df['sampling_rate'].hist(grid=False)
    # plt.title('sampling Rate')
    # plt.savefig('sampling_rate.png')
    # plt.show()
    #
    #
    # df['dendrite_type'].value_counts().plot(kind='bar')
    # plt.title('Dendrite type')
    # plt.show()
    # plt.savefig('dendrite_typs.png')
