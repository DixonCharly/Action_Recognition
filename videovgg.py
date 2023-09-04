import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load the pre-trained model architecture (VGG16-based model)
vgg_model = tf.keras.applications.VGG16(include_top=False, input_shape=(160, 160, 3), pooling='avg')
for layer in vgg_model.layers:
    layer.trainable = False

model = keras.Sequential([
    vgg_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adding dropout layer for regularization
    Dense(256, activation='relu'), # Adding another dense layer
    Dense(15, activation='softmax')
])

# Load the trained weights
model.load_weights("VGG_model_with_extra_layers.h5")

# Define a dictionary for mapping label indices to their corresponding actions
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

# Function to perform action recognition on webcam frames
def recognize_action_from_webcam():
    cap = cv2.VideoCapture(0)  # 0 means the default webcam (you can change it to a different camera ID if available)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (160, 160))
        video_data = np.array([resized_frame])

        # Normalize the pixel values to be between 0 and 1
        video_data = video_data / 255.0

        # Make predictions using the model
        predictions = model.predict(video_data)

        # Get the action with the highest probability
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        predicted_action = label_to_action[predicted_label_index]

        # Display the predicted action on the frame
        cv2.putText(frame, f"Action: {predicted_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the webcam stream
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start recognition from the webcam
recognize_action_from_webcam()

