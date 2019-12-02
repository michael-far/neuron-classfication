import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import fastai.vision as vi
from fastai.vision.data import imagenet_stats
from helper_func import calc_metrics, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold

BATCH_SIZE = 16

db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]

# CLASSES = ['2', '4', '5', '6']
CLASSES = ['spiny', 'aspiny']
n_classes = len(CLASSES)
df['file_name'] = df.index

kf = StratifiedKFold(n_splits=5, random_state=42)
stats = []
for train_index, test_index in kf.split(df, df['dendrite_type']):

    train = df.iloc[train_index]
    test = df.iloc[test_index]


    # data = vi.ImageDataBunch.from_folder('data/images/3dgadf', train="train", valid='test',
    #                                   ds_tfms=None, bs=8)

    train = vi.ImageDataBunch.from_df('data/images/3dgadf', df=train, suffix='.png', valid_pct=0, fn_col='file_name', label_col='dendrite_type')
    train.normalize()
    test = vi.ImageDataBunch.from_df('data/images/3dgadf', df=test, suffix='.png', valid_pct=0,  fn_col='file_name', label_col='dendrite_type')
    test.normalize()
    #
    model = vi.cnn_learner(train, vi.models.resnet50, metrics=vi.accuracy)
    #
    model.fit(epochs=10, lr=0.01)
    #
    model.save('3dgadf_by_layers')
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

sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
plot_confusion_matrix(sum_cm, 'Transfer Learning model')