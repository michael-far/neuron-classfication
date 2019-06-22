import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Dropout, Flatten, Conv2D
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam


from eda import calc_metrics

BATCH_SIZE = 8


db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
n_classes = len(np.unique(df['dendrite_type']))

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
# model.add(Dense(6, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




train_datagen = ImageDataGenerator(validation_split=0.25)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('data/images/3dgadf/train',
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('data/images/3dgadf/test',
                                                         target_size=(224,224),
                                                         color_mode='rgb',
                                                         batch_size=1,
                                                         class_mode='binary',
                                                         shuffle=False,
                                                         subset='validation') # set as validation data



# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
step_size_val= validation_generator.n//validation_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch= step_size_train,
                    epochs=1, validation_data = validation_generator, validation_steps=step_size_val)
model.save('data/models/gadf_cnn')
pred = model.predict_generator(validation_generator, steps=validation_generator.n)
pred = np.where(pred > 0.5, 1.0, 0.0)
calc_metrics(validation_generator.classes, pred)