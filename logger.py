from pymongo import MongoClient
from datetime import datetime

# MongoDB Atlas connection URI
uri = "mongodb+srv://GazeTracker:gazer1@GazeTracker.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

# Accessing the database and collections
db = client['GazeTracker']
eye_collection = db['eye_tracking_data']
click_collection = db['click_events']
calibration_collection = db['calibration_points']



# Insert eye tracking data
def insert_eye_tracking_data(user_id, x, y, calibrated=True):
    doc = {
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "x": x,
        "y": y,
        "calibrated": calibrated
    }
    eye_collection.insert_one(doc)

# Insert click event
def insert_click_event(user_id, event_type):
    doc = {
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "event_type": event_type  # blink, dwell, double_blink
    }
    click_collection.insert_one(doc)

# Insert calibration points
def insert_calibration_points(user_id, points):
    doc = {
        "user_id": user_id,
        "calibration_time": datetime.utcnow(),
        "points": [{"x": p[0], "y": p[1]} for p in points]
    }
    calibration_collection.insert_one(doc)

# Example usage
if __name__ == "__main__":
    insert_eye_tracking_data("user1", 420, 300)
    insert_click_event("user1", "dwell")
    insert_calibration_points("user1", [(100, 100), (500, 300), (960, 540)])