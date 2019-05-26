import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import fastai.vision as vi


BATCH_SIZE = 16


db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
# CLASSES = ['2', '4', '5', '6']
CLASSES = ['spiny', 'aspiny']
n_classes = len(CLASSES)

np.random.seed(42)
data = vi.ImageDataBunch.from_folder('data/images/3dgadf_copy', train="train", valid='test',
                                  ds_tfms=vi.get_transforms(do_flip=False), bs=16)
data.normalize()

learn = vi.cnn_learner(data, vi.models.resnet50, metrics=vi.accuracy)

learn.fit(epochs=100, lr=0.01)

learn.save('3dgadf')
pred = []
for item in data.valid_ds:
        pred.append(str(learn.predict(item[0])[0]))
true = [str(x) for x in data.valid_ds.y]
acc = accuracy_score(true, pred)
c = confusion_matrix(true, pred)
print(c)
print(acc)
