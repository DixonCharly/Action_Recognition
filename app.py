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

app = Flask(__name__, static_folder='static')

# SQLite database configuration
db_path = 'sqlite:///action_log.db'

resnet_model = Sequential()
resnet_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dropout(0.5))
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dense(15, activation='softmax'))

resnet_model.load_weights("resnet_model_with_extra_layers.h5")

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

def create_db_connection():
    conn = sqlite3.connect('action_log.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_log (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            action TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def recognize_action():
    conn, cursor = create_db_connection()
    
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

        # Get current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Insert recognized action and timestamp into the database
        cursor.execute('INSERT INTO action_log (timestamp, action) VALUES (?, ?)', (timestamp, predicted_action))
        conn.commit()

        cv2.putText(frame, f"Action: {predicted_action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_action(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    print(f"Received start_time: {start_time}")
    print(f"Received end_time: {end_time}")

    conn, cursor = create_db_connection()

    # Construct the SQL query
    sql_query = f"SELECT * FROM action_log WHERE timestamp >= ? AND timestamp <= ?;"
    
    # Add debugging output for the SQL query
    print(f"Generated SQL query: {sql_query}")

    # Query the database for activity data within the specified time range
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

if __name__ == '__main__':
    app.run(debug=True)
