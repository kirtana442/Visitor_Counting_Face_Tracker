import cv2
import numpy as np
import os
import json
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

with open("config.json") as f:
    config = json.load(f)

FRAME_SKIP = config["frame_skip"]
SIM_THRESHOLD = config["similarity_threshold"]
EXIT_TIME = config["exit_time_seconds"]

cap = cv2.VideoCapture("video_sample1.mp4")

model = YOLO("yolov8n.pt")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    embedding BLOB
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER,
    event TEXT,
    timestamp TEXT,
    image_path TEXT
)
""")

os.makedirs("logs/entries", exist_ok=True)
os.makedirs("logs/exits", exist_ok=True)

log_file = open("events.log", "a")

known_faces = {}
active_faces = {}
last_exit_time = {}
face_id_counter = 0
COOLDOWN = 5

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def update_embedding(old_emb, new_emb):
    return (old_emb + new_emb) / 2

def get_face_id(embedding):
    global face_id_counter
    best_match = None
    best_score = -1

    for fid, emb in known_faces.items():
        score = cosine_similarity(embedding, emb)
        if score > best_score:
            best_score = score
            best_match = fid

    if best_score > SIM_THRESHOLD:
        known_faces[best_match] = update_embedding(known_faces[best_match], embedding)
        return best_match

    face_id_counter += 1
    fid = face_id_counter
    known_faces[fid] = embedding

    cursor.execute(
        "INSERT INTO faces (id, embedding) VALUES (?, ?)",
        (fid, embedding.tobytes())
    )
    conn.commit()

    return fid

def log_event(fid, event_type, face_img=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = None

    if face_img is not None:
        folder = "entries" if event_type == "ENTRY" else "exits"
        path = f"logs/{folder}/face_{fid}_{timestamp}.jpg"
        cv2.imwrite(path, face_img)

    log_file.write(f"[{timestamp}] {event_type} FaceID={fid} {path}\n")

    cursor.execute(
        "INSERT INTO events (face_id, event, timestamp, image_path) VALUES (?, ?, ?, ?)",
        (fid, event_type, timestamp, path)
    )
    conn.commit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    results = model(frame)[0]

    if len(results.boxes) == 0:
        continue

    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        continue

    faces = app.get(face_crop)
    if len(faces) == 0:
        continue

    embedding = faces[0].embedding
    fid = get_face_id(embedding)

    if fid not in active_faces:
        if fid in last_exit_time and (time.time() - last_exit_time[fid] < COOLDOWN):
            continue
        log_event(fid, "ENTRY", face_crop)

    active_faces[fid] = time.time()

    now = time.time()
    to_remove = []

    for fid_check, last_seen in active_faces.items():
        if now - last_seen > EXIT_TIME:
            log_event(fid_check, "EXIT")
            last_exit_time[fid_check] = time.time()
            to_remove.append(fid_check)

    for fid_check in to_remove:
        del active_faces[fid_check]

now = time.time()

for fid in list(active_faces.keys()):
    log_event(fid, "EXIT")
    last_exit_time[fid] = now

active_faces.clear()

cap.release()
log_file.close()
conn.close()