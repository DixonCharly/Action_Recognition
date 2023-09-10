from flask import Flask, render_template, Response  # Import Flask for web application
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
from tensorflow.keras.applications import ResNet50  # Import ResNet50 model
from tensorflow.keras.layers import Dense, Flatten, Dropout  # Import Keras layers
from tensorflow.keras.models import Sequential  # Import Keras sequential model

# Create a Flask app
app = Flask(__name__, static_folder='static')

# Define the ResNet50-based model architecture
resnet_model = Sequential()
resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))  # Pre-trained ResNet50 base
resnet_model.add(Flatten())  # Flatten layer for feature vector
resnet_model.add(Dense(512, activation='relu'))  # Dense layer with ReLU activation
resnet_model.add(Dropout(0.5))  # Dropout layer for regularization
resnet_model.add(Dense(256, activation='relu'))  # Another dense layer with ReLU activation
resnet_model.add(Dense(15, activation='softmax'))  # Output layer with softmax activation for 15 classes

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
def recognize_action():
    cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:  # If frame reading fails, exit the loop
            break

        resized_frame = cv2.resize(frame, (160, 160))  # Resize the frame for model input
        img = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize pixel values to [0, 1]

        predictions = resnet_model.predict(img)  # Make predictions using the model
        predicted_label_index = np.argmax(predictions, axis=1)[0]  # Get the predicted label index
        predicted_action = label_to_action[predicted_label_index]  # Map label index to action name

        # Add the predicted action as text to the frame
        cv2.putText(frame, f"Action: {predicted_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)  # Convert the frame to JPEG format for streaming
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close OpenCV windows

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template for the home page

# Route to the video feed
@app.route('/video_feed')
def video_feed():
    return Response(recognize_action(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Stream frames as multipart responses

if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask app in debug mode if the script is executed directly
