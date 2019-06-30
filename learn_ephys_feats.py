import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from eda import calc_metrics, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold

BATCH_SIZE = 64


db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df = df.dropna(axis=1)
df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
df['dendrite_type'] = pd.Categorical(df['dendrite_type'])
df['dendrite_type'] = df['dendrite_type'].cat.codes
irrlevent_columns = [c for c in df.columns if c.endswith('index')] +\
                    ['layer', 'mean_clipped', 'structure_area_abbrev', 'sampling_rate']
# irrlevent_columns = [c for c in df.columns if c.endswith('index')] +\
#                     ['dendrite_type', 'mean_clipped', 'structure_area_abbrev', 'sampling_rate']
df = df.drop(irrlevent_columns, axis=1)
y = df.pop('dendrite_type')
y = y.values.astype(float)
# y = df.pop('layer')
x = df.values
# pca_plot(x, y)
x = scale(x)
# x_train, x_test ,y_train, y_test = train_test_split(x, y, train_size=0.75)
# y_train_numeric = y_train
# y_train = to_categorical(y_train.values)


n_feats = len(df.columns)
kf = StratifiedKFold(n_splits=5, random_state=42)
stats = []
for train_index, test_index in kf.split(x, y):
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=n_feats))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(y_train.shape[1],  activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100)
    model.save('data/models/ephys_dnn')
    pred = model.predict_classes(x_test)
    results = calc_metrics(y_test, pred)
    stats.append(results)
mean_f1 = np.asarray([x[0] for x in stats]).mean()
mean_accuracy = np.asarray([x[1] for x in stats]).mean()
print('Accuracy: {}, f1: {}'.format(mean_accuracy, mean_f1))

sum_cm = np.asarray([x[2] for x in stats]).sum(axis=0)
sum_cm = sum_cm / sum_cm.astype(np.float).sum(axis=1)
plot_confusion_matrix(sum_cm, 'DNN model')