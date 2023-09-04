import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setting up the path and loading csv files
#"D:/PY/PY ML/Human_Action_Recognition/Training_set.csv"

train_csv_path = "D:/PY/Action_Recognition/Human_Action_Recognition/Training_set.csv"
test_csv_path = "D:/PY/Action_Recognition/Human_Action_Recognition/Testing_set.csv"
train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)

# Data Preprocessing and Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Create train and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="D:/PY/Action_Recognition/Human_Action_Recognition/train/",
    x_col="filename",
    y_col="label",
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="D:/PY/Action_Recognition/Human_Action_Recognition/train/",
    x_col="filename",
    y_col="label",
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Create ResNet model with additional layers
resnet_model = keras.Sequential()
resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dropout(0.5))  # Adding dropout layer for regularization
resnet_model.add(Dense(256, activation='relu'))  # Adding another dense layer
resnet_model.add(Dense(15, activation='softmax'))

resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model.summary()

# Training the model
history = resnet_model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Save the model weights
resnet_model.save_weights("resnet_model_with_extra_layers.h5")
