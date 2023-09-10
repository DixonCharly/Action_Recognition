import numpy as np  # Import NumPy for numerical operations
from PIL import Image  # Import Python Imaging Library for image handling
import matplotlib.pyplot as plt  # Import Matplotlib for image display
import tensorflow as tf  # Import TensorFlow for deep learning
from keras.layers import Dense, Flatten, Dropout  # Import Keras layers for model construction
from keras.models import Model, Sequential  # Import Keras Model and Sequential for model creation
from keras.applications.vgg16 import VGG16  # Import VGG16 model from Keras applications

# Create a new instance of the VGG16 model
vgg_model = Sequential()

# Load a pre-trained VGG16 model with custom modifications
pretrained_model = tf.keras.applications.VGG16(
    include_top=False,  # Exclude fully connected layers
    input_shape=(160, 160, 3),  # Input image shape
    pooling='avg',  # Global average pooling for the last layer
    classes=15,  # Number of classes for the output layer
    weights='imagenet'  # Load pre-trained weights from ImageNet
)

# Freeze all layers in the pre-trained model (prevent training)
for layer in pretrained_model.layers:
    layer.trainable = False

# Add the pre-trained model to the custom VGG model
vgg_model.add(pretrained_model)  # Add the pre-trained model as a layer
vgg_model.add(Flatten())  # Flatten the output for fully connected layers
vgg_model.add(Dense(512, activation='relu'))  # Add a dense layer with ReLU activation
vgg_model.add(Dropout(0.5))  # Add a dropout layer for regularization
vgg_model.add(Dense(256, activation='relu'))  # Add another dense layer with ReLU activation
vgg_model.add(Dense(15, activation='softmax'))  # Add an output layer with softmax activation

# Load saved weights into the model
vgg_model.load_weights('VGG_model_with_extra_layers.h5')  # Load pre-trained model weights

# Function to read images as arrays
def read_image(fn):
    image = Image.open(fn)  # Open an image file using PIL
    return np.asarray(image.resize((160, 160)))  # Resize and convert to a NumPy array

# Function to predict the action class of a test image
def test_predict(test_image):
    result = vgg_model.predict(np.asarray([read_image(test_image)]))  # Predict using the model

    # Find the index of the predicted class with the highest probability
    itemindex = np.where(result == np.max(result))
    prediction = itemindex[1][0]
    return prediction  # Return the predicted class index

# Dictionary mapping class numbers to action labels
label_to_action = {
    0: "sitting",
    1: "using laptop",
    2: "hugging",
    3: "sleeping",
    4: "drinking",
    5: "clapping",
    6: "dancing",
    7: "cycling",
    8: "calling",
    9: "laughing",
    10: "eating",
    11: "fighting",
    12: "listening_to_music",
    13: "running",
    14: "texting"
}

# Test image prediction (replace with your test image path)
test_image = './Human_Action_Recognition/test/Image_1454.jpg'
predicted_class = test_predict(test_image)  # Predict the action class of the test image

# Display the test image and the predicted action label
test_image = Image.open(test_image)  # Open the test image using PIL
plt.imshow(test_image)  # Display the test image
plt.title(f"Predicted action: {label_to_action[predicted_class]}")  # Display the predicted action label
plt.show()  # Show the image and title
