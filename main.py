import cv2
import os
import json
import time
from datetime import datetime
import math

# Load config
with open("config.json") as f:
    config = json.load(f)

FRAME_SKIP = config["frame_skip"]
DIST_THRESHOLD = config["distance_threshold"]

# Setup
cap = cv2.VideoCapture("video_sample1.mp4")  
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
os.makedirs("logs/entries", exist_ok=True)
os.makedirs("logs/exits", exist_ok=True)

log_file = open("events.log", "a")

# Tracking
face_id_counter = 0
active_faces = {}  # id -> (x, y, last_seen)

def get_center(x, y, w, h):
    return (x + w//2, y + h//2)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_centers = []

    for (x, y, w, h) in faces:
        center = get_center(x, y, w, h)
        current_centers.append((center, (x, y, w, h)))

    matched_ids = set()

    for center, box in current_centers:
        matched = False

        for fid in list(active_faces.keys()):
            prev_center = active_faces[fid][0]

            if distance(center, prev_center) < DIST_THRESHOLD:
                active_faces[fid] = (center, time.time())
                matched_ids.add(fid)
                matched = True
                break

        if not matched:
            # NEW FACE → ENTRY
            face_id_counter += 1
            fid = face_id_counter
            active_faces[fid] = (center, time.time())

            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"logs/entries/face_{fid}_{timestamp}.jpg"
            cv2.imwrite(path, face_img)

            log_file.write(f"[{timestamp}] ENTRY FaceID={fid} {path}\n")
            print(f"ENTRY FaceID={fid}")

    # Detect exits
    current_time = time.time()
    to_remove = []

    for fid in active_faces:
        last_seen = active_faces[fid][1]

        if current_time - last_seen > 2:  # disappeared
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            log_file.write(f"[{timestamp}] EXIT FaceID={fid}\n")
            print(f"EXIT FaceID={fid}")

            to_remove.append(fid)

    for fid in to_remove:
        del active_faces[fid]

cap.release()
log_file.close()