from flask import Flask, render_template, Response  # Import Flask 
import cv2  # Import OpenCV for image processing
import tensorflow as tf  # Import TensorFlow for deep learning
import numpy as np  # Import NumPy 
from tensorflow import keras  # Import Keras for training neural networks
from tensorflow.keras.layers import Flatten, Dense, Dropout  # Import Keras layers

# Load a pre trained VGG16-based model for feature extraction
vgg_model = tf.keras.applications.VGG16(include_top=False, input_shape=(160, 160, 3), pooling='avg')

# Freezing all layers in the VGG model to prevent further training
for layer in vgg_model.layers:
    layer.trainable = False

# Create a new sequential model by stacking layers
model = keras.Sequential([
    vgg_model,  # Feature extraction
    Flatten(),  # Flatten the output
    Dense(512, activation='relu'),  # Dense layer with ReLU activation
    Dropout(0.5),  # Dropout layer for regularization
    Dense(256, activation='relu'),  # Another dense layer with ReLU activation
    Dense(15, activation='softmax')  # Output layer with softmax activation for 15 classes
])

# Load pre-trained weights into the model
model.load_weights("VGG_model_with_extra_layers.h5")

# Define a mapping from label indices to action names
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

# Function for perform action recognition on webcam frames
def recognize_action():
    cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
   # frame_count = 0
   # start_time = time.time()

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:  # If frame reading fails, exit the loop
            break

        resized_frame = cv2.resize(frame, (160, 160))  # Resize the frame for model input
        video_data = np.array([resized_frame])  # Convert the frame to a NumPy array
        video_data = video_data / 255.0  # Normalize pixel values to [0, 1]

        predictions = model.predict(video_data)  # Make predictions using the model
        predicted_label_index = np.argmax(predictions, axis=1)[0]  # Get the predicted label index
        predicted_action = label_to_action[predicted_label_index]  # Map label index to action name

        # Add the predicted action as text to the frame
        cv2.putText(frame, f"Action: {predicted_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)  # Convert the frame to JPEG format for streaming
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

      #  frame_count += 1
       # if frame_count % 10 == 0:  # Calculate and print FPS every 10 frames
        #    elapsed_time = time.time() - start_time
         #   fps = frame_count / elapsed_time
          #  print(f"FPS: {fps:.2f}")

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close OpenCV windows

# Create the Flask app
app = Flask(__name__, static_folder='static')

# Define a route to the home page
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template for the home page

# Define a route to the video feed
@app.route('/video_feed')
def video_feed():
    return Response(recognize_action(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Stream frames as multipart responses

if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask app in debug mode if the script is executed directly
