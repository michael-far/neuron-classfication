import pandas as pd
import numpy as np


from fastai import *
from fastai.vision import *

BATCH_SIZE = 8


db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
n_classes = len(np.unique(df['layer']))

np.random.seed(42)
data = ImageDataBunch.from_folder('data/images/3dgaf_copy', train="train", valid='test',
        ds_tfms=None, num_workers=4, bs=8).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.unfreeze()
learn.lr_find()

learn.fit_one_cycle(10, max_lr=slice(1e-6, 3e-5))
learn.save('stage0_GAFD')
