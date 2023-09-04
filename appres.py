from flask import Flask, render_template, Response
import cv2
import numpy as np
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

#resnet_model = Sequential()
#resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
#resnet_model.add(Dense(15, activation='softmax'))


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
def recognize_action():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (160, 160))
        img = np.expand_dims(resized_frame, axis=0)
        img = img / 255.0

        predictions = resnet_model.predict(img)
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