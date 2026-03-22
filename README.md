# Intelligent Face Tracker with Auto Registration and Visitor Counting

## Overview

This project implements an AI-driven face tracking and visitor counting system. It processes a video stream to detect faces, generate embeddings, assign unique identities, and log entry and exit events. The system maintains a count of unique visitors by re-identifying previously seen faces using facial embeddings.

The implementation is designed to be modular and extensible, allowing integration with production-grade tracking systems and real-time camera streams.

---
### Project Explanation Video

[Watch the demo video](https://drive.google.com/file/d/11bKmJCFeGQ7BamsNUB5bl0JOjOEh759p/view?usp=sharing)
---

## Features

* Face detection using YOLOv8
* Face recognition using InsightFace embeddings
* Automatic registration of new faces with unique IDs
* Re-identification using cosine similarity
* Entry and exit event detection
* Logging to:

  * Local filesystem (cropped face images)
  * Log file (`events.log`)
  * SQLite database (`faces.db`)
* Configurable parameters via `config.json`
* Unique visitor counting based on registered identities

---

## System Architecture

Pipeline:

Video Input → YOLOv8 Detection → Face Cropping → InsightFace Embedding
→ Identity Matching (Cosine Similarity) → ID Assignment
→ Tracking (Active Face Memory) → Entry/Exit Logging → Database Storage

---

## Directory Structure

```
face_tracker/
│── main.py
│── config.json
│── video.mp4
│── faces.db
│── events.log
│── logs/
│    ├── entries/
│    └── exits/
```

---

## Setup Instructions

### 1. Install Dependencies

```
pip install ultralytics insightface onnxruntime opencv-python numpy
```

### 2. Add Input Video

Place your test video in the project directory and name it:

```
video.mp4
```

### 3. Run the Application

```
python main.py
```

---

## Configuration

`config.json`

```
{
  "frame_skip": 5,
  "similarity_threshold": 0.5,
  "exit_time_seconds": 3
}
```

### Parameters

* `frame_skip`: Number of frames skipped between detection cycles
* `similarity_threshold`: Threshold for cosine similarity to match identities
* `exit_time_seconds`: Time after which a face is considered to have exited

---

## Logging System

### Filesystem Logging

* Entry images stored in: `logs/entries/`
* Exit images stored in: `logs/exits/`

### Log File

`events.log` records:

* Face entry events
* Face exit events
* Assigned face IDs
* Timestamps

### Database

SQLite database (`faces.db`) contains:

#### Table: faces

* `id`: Unique face identifier
* `embedding`: Stored facial embedding

#### Table: events

* `face_id`
* `event` (ENTRY / EXIT)
* `timestamp`
* `image_path`

---

## Unique Visitor Counting

* Each new face is assigned a unique ID
* Re-identification is performed using cosine similarity on embeddings
* The number of unique visitors corresponds to the number of unique IDs stored in the database
* Repeated appearances of the same individual do not increase the unique visitor count

---

## Assumptions

* Each detected face produces a valid embedding
* Cosine similarity is sufficient for identity matching
* Video input is reasonably clear for face detection
* Faces remain visible long enough for embedding extraction

---

## Limitations

* Tracking is implemented using time-based presence instead of advanced trackers such as DeepSort or ByteTrack
* CPU-based inference may not achieve real-time performance for high-resolution streams
* Recognition accuracy depends on lighting, pose, and similarity threshold tuning
* RTSP stream handling and reconnection logic are not implemented

---

## Compute Considerations

* YOLOv8 inference: CPU-bound, moderate usage
* InsightFace embedding generation: CPU-intensive
* Memory usage increases with number of registered faces
* GPU acceleration can significantly improve performance if enabled

---

## Note

This project is a part of a hackathon run by https://katomaran.com
