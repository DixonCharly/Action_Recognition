import os  # Import os Module
import glob  # Import glob for file matching
import numpy as np  # Import NumPy 
import pandas as pd  # Import Pandas for data manipulation
import tensorflow as tf  # Import TensorFlow for deep learning
from tensorflow import keras  # Import Keras for building deep learning models
from tensorflow.keras.applications import ResNet50  # Import the ResNet50 model
from tensorflow.keras.layers import Flatten, Dense, Dropout  # Import Keras layers for model construction
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation

# Setting up the path and loading CSV files
train_csv_path = "D:/PY/Action_Recognition/Human_Action_Recognition/Training_set.csv"  # Path to the training CSV file
test_csv_path = "D:/PY/Action_Recognition/Human_Action_Recognition/Testing_set.csv"  # Path to the testing CSV file
train_csv = pd.read_csv(train_csv_path)  # Load the training CSV file using Pandas
test_csv = pd.read_csv(test_csv_path)  # Load the testing CSV file using Pandas

# Data Preprocessing and Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Rescale pixel values to [0, 1]
    validation_split=0.2,  # Split data into validation set (20%)
    horizontal_flip=True,  # Horizontal flip for data augmentation
    rotation_range=20,  # Rotation range for data augmentation
    width_shift_range=0.2,  # Width shift range for data augmentation
    height_shift_range=0.2  # Height shift range for data augmentation
)

# Create train and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="D:/PY/Action_Recognition/Human_Action_Recognition/train/",  # Directory containing training images
    x_col="filename",  # Column name for image file names
    y_col="label",  # Column name for image labels
    target_size=(160, 160),  # Target image size
    batch_size=32,  # Batch size for training
    class_mode="categorical",  # Categorical mode for multi-class classification
    subset="training"  # Subset for training data
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="D:/PY/Action_Recognition/Human_Action_Recognition/train/",  # Directory containing training images
    x_col="filename",  # Column name for image file names
    y_col="label",  # Column name for image labels
    target_size=(160, 160),  # Target image size
    batch_size=32,  # Batch size for training
    class_mode="categorical",  # Categorical mode for multi-class classification
    subset="validation"  # Subset for validation data
)

# Create ResNet model with additional layers
resnet_model = keras.Sequential()
resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))  # Add ResNet50 base model
resnet_model.add(Flatten())  # Flatten the output from the base model
resnet_model.add(Dense(512, activation='relu'))  # Add a dense layer with ReLU activation
resnet_model.add(Dropout(0.5))  # Adding dropout layer for regularization
resnet_model.add(Dense(256, activation='relu'))  # Adding another dense layer
resnet_model.add(Dense(15, activation='softmax'))  # Add an output layer with softmax activation

# Compile the ResNet model
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
resnet_model.summary()

# Train the model
history = resnet_model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Save the model weights
resnet_model.save_weights("resnet_model_with_extra_layers.h5")
