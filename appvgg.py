from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
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
def recognize_action():
    cap = cv2.VideoCapture(0)
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

        cv2.putText(frame, f"Action: {predicted_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Create the Flask app
app = Flask(__name__, static_folder='static')

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to the video feed
@app.route('/video_feed')
def video_feed():
    return Response(recognize_action(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
