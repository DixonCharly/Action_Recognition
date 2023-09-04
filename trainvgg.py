import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setting up the path and loading csv files
train_csv = pd.read_csv("./Human_Action_Recognition/Training_set.csv")
test_csv = pd.read_csv("./Human_Action_Recognition/Testing_set.csv")

# Data Preprocessing and Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,          # Normalize pixel values between 0 and 1
    validation_split=0.2,        # Split data into training and validation sets
    horizontal_flip=True,        # Randomly flip images horizontally
    rotation_range=20,           # Randomly rotate images
    width_shift_range=0.2,       # Randomly shift images horizontally
    height_shift_range=0.2       # Randomly shift images vertically
)

# Create train and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="./Human_Action_Recognition/train/",
    x_col="filename",
    y_col="label",
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="./Human_Action_Recognition/train/",
    x_col="filename",
    y_col="label",
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build a custom model with additional layers
input_layer = keras.Input(shape=(160, 160, 3))
vgg_model = tf.keras.applications.VGG16(include_top=False, input_tensor=input_layer, pooling='avg')
for layer in vgg_model.layers:
    layer.trainable = False

custom_model = keras.Sequential([
    input_layer,
    vgg_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adding dropout layer for regularization
    Dense(256, activation='relu'),  # Adding another dense layer
    Dense(15, activation='softmax')
])

model = custom_model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(train_generator, epochs=60, validation_data=validation_generator)

# Save the model weights
model.save_weights("VGG_model_with_extra_layers.h5")
