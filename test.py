import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16

# Create a new instance of the VGG16 mode
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dropout(0.5))  # Adding dropout layer for regularization
vgg_model.add(Dense(256, activation='relu')) # Adding another dense layer
vgg_model.add(Dense(15, activation='softmax'))

# Load the saved weights into the model
vgg_model.load_weights('VGG_model_with_extra_layers.h5')

# Function to read images as array

def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))

# Function to predict

def test_predict(test_image):
    result = vgg_model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
#     print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)
    return prediction

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

# Test image prediction
#test_image = './Human_Action_Recognition/test/Image_16.jpg'
test_image = './Human_Action_Recognition/test/Image_1454.jpg'
#test_image = './Human_Action_Recognition/test/Image_574.jpg'
#test_image = './Human_Action_Recognition/test/Image_275.jpg'

predicted_class = test_predict(test_image)


# Display the test image
test_image = Image.open(test_image)
plt.imshow(test_image)
plt.title(f"Predicted action: {label_to_action[predicted_class]}")
plt.show()
