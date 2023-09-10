import os  # Import the os module for file and directory operations
import glob  # Import glob for file matching
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import tensorflow as tf  # Import TensorFlow for deep learning
from tensorflow import keras  # Import Keras for building deep learning models
from tensorflow.keras.layers import Flatten, Dense, Dropout  # Import Keras layers for model construction
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation

# Setting up the path and loading csv files
train_csv = pd.read_csv("./Human_Action_Recognition/Training_set.csv")  # Load the training CSV file
test_csv = pd.read_csv("./Human_Action_Recognition/Testing_set.csv")  # Load the testing CSV file

# Data Preprocessing and Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values between 0 and 1
    validation_split=0.2,  # Split data into training and validation sets (80% training, 20% validation)
    horizontal_flip=True,  # Randomly flip images horizontally for data augmentation
    rotation_range=20,  # Randomly rotate images by up to 20 degrees for data augmentation
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of the image width
    height_shift_range=0.2  # Randomly shift images vertically by up to 20% of the image height
)

# Create train and validation generators for image data
train_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="./Human_Action_Recognition/train/",  # Directory containing training images
    x_col="filename",  # Column name for image file names
    y_col="label",  # Column name for image labels
    target_size=(160, 160),  # Target image size (rescaled to 160x160 pixels)
    batch_size=32,  # Batch size for training
    class_mode="categorical",  # Categorical mode for multi-class classification
    subset="training"  # Subset for training data
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory="./Human_Action_Recognition/train/",  # Directory containing training images
    x_col="filename",  # Column name for image file names
    y_col="label",  # Column name for image labels
    target_size=(160, 160),  # Target image size (rescaled to 160x160 pixels)
    batch_size=32,  # Batch size for training
    class_mode="categorical",  # Categorical mode for multi-class classification
    subset="validation"  # Subset for validation data
)

# Build a custom model with additional layers on top of a pre-trained VGG16 base
input_layer = keras.Input(shape=(160, 160, 3))  # Define the input shape for images
vgg_model = tf.keras.applications.VGG16(include_top=False, input_tensor=input_layer, pooling='avg')  # Load VGG16 base model
for layer in vgg_model.layers:
    layer.trainable = False  # Freeze the layers of the pre-trained VGG16 base

# Construct the custom model by stacking layers
custom_model = keras.Sequential([
    input_layer,  # Input layer
    vgg_model,  # Pre-trained VGG16 base
    Flatten(),  # Flatten the output from the base model
    Dense(512, activation='relu'),  # Add a dense layer with ReLU activation
    Dropout(0.5),  # Adding dropout layer for regularization
    Dense(256, activation='relu'),  # Adding another dense layer
    Dense(15, activation='softmax')  # Add an output layer with softmax activation for 15 classes
])

model = custom_model  # Assign the custom model to the variable "model"
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
model.summary()  # Display a summary of the model architecture

# Train the model on the training data and validate on the validation data
history = model.fit(train_generator, epochs=60, validation_data=validation_generator)

# Save the trained model weights to a file
model.save_weights("VGG_model_with_extra_layers.h5")
