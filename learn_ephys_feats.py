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
from eda import pca_plot

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
# y = to_categorical(y.values)
x = df.values
# pca_plot(x, y)
x = scale(x)
x_train, x_test ,y_train, y_test = train_test_split(x, y, train_size=0.75)

n_feats = len(df.columns)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=n_feats))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

pred = model.predict_classes(x_test)
print(accuracy_score(y_test, pred))
