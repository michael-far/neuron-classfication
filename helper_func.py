from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# db_file = '/media/wd/data/cells/db.p'
# df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
# df = pd.read_pickle(db_file)
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


def arrange_files(out_dir, ext, class_type):
    cells = df.index[df['dendrite_type'].isin(['spiny', 'aspiny'])]
    for cell in cells:
        # convert_cell(cell, df, out_dir)
        move_to_class_folder(cell, class_type, out_dir, ext)
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


def calc_metrics(y_true, y_pred):
    if type(y_true[0]) == str:
        f1 = f1_score(y_true, y_pred, pos_label='aspiny')
    else:
        f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    confuse = confusion_matrix(y_true, y_pred)
    return f1, accuracy, confuse

def random_search_helper(params: Dict[str, List], n_iters:int = 10) -> dict:
    prev_chosen = []
    i = 0
    while i < n_iters:
        res = dict.fromkeys(params.keys())
        for k, v in params.items():
            res[k] = np.random.choice(v)
        if not any([res == x for x in prev_chosen]):
            i += 1
            prev_chosen.append(res)
            yield res


def plot_confusion_matrix(cm, task, cmap=plt.cm.Blues):
    title = 'Confusion matrix: ' + task
    # Only use the labels that appear in the data
    classes = ['Spiny', 'Aspiny']

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.show()


def show_rand_gadf(src_dir):
    import matplotlib.image as mpimg
    files = [f for f in os.listdir(src_dir)]
    files_to_show = random.sample(files, 6)
    f, axes = plt.subplots(2, 2)
    for ind, ax in enumerate(axes.reshape(-1)):
        img = mpimg.imread('{}/{}'.format(src_dir, files_to_show[ind]))
        ax.imshow(img)
        ax.axis('off')
        cell_id = files_to_show[ind].split('.')[0]
        ax.set_title(df.loc[cell_id]['dendrite_type'])

    plt.suptitle('Example of GAF images')
    plt.show()




if __name__ == '__main__':
    out_dir = 'data/time_series/'
    # show_rand_gadf('data/images/3dgadf')
    # arrange_files(out_dir, ext='npy', class_type='dendrite_type')
    #
    df['layer'] = df['layer'].replace(['6a', '6b'], 6)
    df['layer'] = df['layer'].replace('2/3', 2)
    df['layer'] = df['layer'].astype('float')

    df['layer'].hist(grid=False)
    plt.title('Layers')
    plt.show()
    plt.savefig('layers.png')

    df['structure_area_abbrev'].value_counts().plot(kind='bar')
    plt.title('Brain regions')
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig('regions.png')
    plt.show()

    df['sampling_rate'] = df['sampling_rate'].astype('float')
    df['sampling_rate'].hist(grid=False)
    plt.title('Sampling rate')
    plt.savefig('sampling_rate.png')
    plt.show()


    df['dendrite_type'].value_counts().plot(kind='bar')
    plt.title('Dendrite type')
    plt.xticks(rotation=0)
    plt.show()
    plt.savefig('dendrite_typs.png')
