import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Dropout, Flatten
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

BATCH_SIZE = 32
DATA = '3dgadf'
# CLASSES = ['2', '4', '5', '6']
CLASSES = ['spiny', 'aspiny']

db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
n_classes = len(CLASSES)

base_model = ResNet50(weights='imagenet',include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
# x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(n_classes, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=preds)


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('data/images/' + DATA,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    # class_mode='categorical',
                                                    classes=CLASSES,
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('data/images/' + DATA,
                                                         target_size=(224,224),
                                                         color_mode='rgb',
                                                         batch_size=BATCH_SIZE,
                                                         # class_mode='categorical',
                                                         classes=CLASSES,
                                                         shuffle=True,
                                                         subset='validation')


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10,validation_data = validation_generator,
                    validation_steps = validation_generator.samples // train_generator.batch_size)