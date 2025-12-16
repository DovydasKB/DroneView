from __future__ import annotations

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

MEDIAMTX_BIN = HERE / ("mediamtx.exe" if os.name == "nt" else "mediamtx")
CFG_PATH = HERE / "mediamtx.generated.yml"

RTMP_PORT = 1935
HLS_PORT = 8888
WEB_PORT = 5000
RTSP_URL = "rtsp://127.0.0.1:8554/live/stream"

MEDIAMTX_CONFIG = f"""
logLevel: info

rtmp: yes
rtmpAddress: :{RTMP_PORT}

hls: yes
hlsAddress: :{HLS_PORT}
hlsAlwaysRemux: yes

rtsp: yes

paths:
  "~^.*$":
    source: publisher
""".lstrip()

app = FastAPI()
mediamtx_proc = None

model = YOLO("yolov8n.pt")

def detection_stream():
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("ERROR: Cannot open RTSP stream")
        return

    last_detect = 0
    DETECT_INTERVAL = 1.0

    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        now = time.time()

        if now - last_detect > DETECT_INTERVAL:
            results = model(frame, conf=0.4, classes=[0], verbose=False)
            detections = results[0].boxes
            last_detect = now
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "HUMAN",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg.tobytes()
            + b"\r\n"
        )


@app.get("/detect")
def detect():
    return StreamingResponse(
        detection_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(f"""
<!doctype html>
<html>
<head>
  <title>Drone Live + Human Detection</title>
  <style>
    body {{ font-family: Arial; margin: 20px; }}
    img {{ max-width: 100%; border: 2px solid #333; }}
  </style>
</head>
<body>
  <h2>Human Detection (LIVE)</h2>
  <p>No buffering. Real frames. ~1s detection latency.</p>
  <img src="/detect">
</body>
</html>
""")

def start_mediamtx():
    global mediamtx_proc

    CFG_PATH.write_text(MEDIAMTX_CONFIG)

    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        preexec_fn = os.setsid

    mediamtx_proc = subprocess.Popen(
        [str(MEDIAMTX_BIN), str(CFG_PATH)],
        stdout=sys.stdout,
        stderr=sys.stderr,
        creationflags=creationflags,
        preexec_fn=preexec_fn,
    )


def stop_mediamtx():
    global mediamtx_proc
    if mediamtx_proc:
        mediamtx_proc.terminate()
        mediamtx_proc.wait()
        mediamtx_proc = None


def main():
    start_mediamtx()

    def cleanup(*_):
        stop_mediamtx()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n=== READY ===")
    print(f"RTMP ingest: rtmp://YOUR_IP:{RTMP_PORT}/live/stream")
    print(f"Detection UI: http://127.0.0.1:{WEB_PORT}\n")

    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT)


if __name__ == "__main__":
    main()
