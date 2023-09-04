import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras.models import Sequential

# Load the model architecture (ResNet50-based model)
resnet_model = Sequential()
resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dropout(0.5))
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dense(15, activation='softmax'))

# Compile the model (if needed)
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the saved model weights
resnet_model.load_weights("resnet_model_with_extra_layers.h5")

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
        img = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize pixel values between 0 and 1

        # Make predictions using the model
        predictions = resnet_model.predict(img)

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

