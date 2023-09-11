import sqlite3
import matplotlib.pyplot as plt
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('activity_recognition.db')
cursor = conn.cursor()

# Fetch all activities from the database
cursor.execute("SELECT activity FROM activities")
activities = cursor.fetchall()

# Extract recognized activities
recognized = [activity[0] for activity in activities]

# Count the occurrences of each activity
activity_counts = dict(zip(*np.unique(recognized, return_counts=True)))

# Create a bar chart to display activity counts
plt.figure(figsize=(10, 6))
plt.bar(activity_counts.keys(), activity_counts.values(), color='blue')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.title('Recognized Activity Counts')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the chart to a file
plt.savefig('activity_counts_chart.png')

# Close the database connection
conn.close()
