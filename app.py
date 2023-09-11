# Import necessary libraries and modules
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
import time
import json
import sqlite3
from datetime import datetime, timedelta

# Create a Flask web application instance
app = Flask(__name__, static_folder='static')

# Specify the path for the SQLite database
db_path = 'sqlite:///action_log.db'

# Create a Sequential model for action recognition using ResNet50
resnet_model = Sequential()

# Add the ResNet50 model as the base layer (no top classification layer)
resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

# Flatten the output of ResNet50
resnet_model.add(Flatten())

# Add an additional fully connected layer with 512 neurons and ReLU activation
resnet_model.add(Dense(512, activation='relu'))

# Add dropout layer to prevent overfitting
resnet_model.add(Dropout(0.5))

# Add another fully connected layer with 256 neurons and ReLU activation
resnet_model.add(Dense(256, activation='relu'))

# Add the final output layer with 15 neurons (corresponding to action classes) and softmax activation
resnet_model.add(Dense(15, activation='softmax'))

# Load pre-trained weights for the model
resnet_model.load_weights("resnet_model_with_extra_layers.h5")

# Define a mapping of action labels to action names
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

# Function to create a database connection
def create_db_connection():
    # Establish a connection to the SQLite database
    conn = sqlite3.connect('action_log.db')
    # Create a cursor object for executing SQL queries
    cursor = conn.cursor()
    # Create the 'action_log' table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_log (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            action TEXT
        )
    ''')
    # Commit the changes to the database
    conn.commit()
    # Return the connection and cursor objects
    return conn, cursor

# Function for recognizing actions from video feed
def recognize_action():
    # Create a database connection
    conn, cursor = create_db_connection()
    
    # Capture video from the default camera (camera index 0)
    cap = cv2.VideoCapture(0)
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 160x160 pixels
        resized_frame = cv2.resize(frame, (160, 160))
        img = np.expand_dims(resized_frame, axis=0)
        img = img / 255.0

        # Make predictions using the ResNet50-based model
        predictions = resnet_model.predict(img)
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        predicted_action = label_to_action[predicted_label_index]

        # Get the current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Insert recognized action and timestamp into the database
        cursor.execute('INSERT INTO action_log (timestamp, action) VALUES (?, ?)', (timestamp, predicted_action))
        conn.commit()

        # Display the predicted action on the frame
        cv2.putText(frame, f"Action: {predicted_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame as JPEG for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

# Define the route for the default homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(recognize_action(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define an API endpoint for retrieving activity data within a specified time range
@app.route('/get_activity_data', methods=['GET'])
def get_activity_data():
    # Retrieve the start and end times for the desired time range
    start_time_str = request.args.get('start_time')  # Format: "YYYY-MM-DDTHH:mm"
    end_time_str = request.args.get('end_time')  # Format: "YYYY-MM-DDTHH:mm"

    # Check if the received timestamps are None (not provided)
    if start_time_str is None or end_time_str is None:
        return jsonify({"error": "Invalid timestamp format"}), 400

    # Convert the received timestamps to the expected format
    start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")

    # Format the timestamps as "YYYY-MM-DD HH:mm:ss"
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
# Add debugging output
  #  print(f"Received start_time: {start_time}")
  #  print(f"Received end_time: {end_time}")

    # Create a database connection
    conn, cursor = create_db_connection()

    # Construct the SQL query to retrieve activity data within the specified time range
    sql_query = f"SELECT * FROM action_log WHERE timestamp >= ? AND timestamp <= ?;"
    
 # Add debugging output for the SQL query
   # print(f"Generated SQL query: {sql_query}")


    # Query the database for activity data
    cursor.execute(sql_query, (start_time, end_time))
    activity_data = cursor.fetchall()
    conn.close()

    # Process the retrieved data to generate patterns
    action_counts = {}  # Dictionary to store action counts

    for entry in activity_data:
        action = entry[2]  # Extract the action from the entry
        action_counts[action] = action_counts.get(action, 0) + 1

    # Create a list of action count entries for the pattern
    pattern = [{"action": action, "count": count} for action, count in action_counts.items()]

    # Return the generated pattern as a JSON response
    response_data = {"pattern": pattern}

    return jsonify(response_data)

# Run the Flask application if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
