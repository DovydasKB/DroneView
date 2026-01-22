from __future__ import annotations

import os
import sys
import subprocess
import time
import threading
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import cv2
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DroneApp")

HERE = Path(__file__).resolve().parent
MEDIAMTX_BIN = HERE / ("mediamtx.exe" if os.name == "nt" else "mediamtx")
CFG_PATH = HERE / "mediamtx.generated.yml"

RTMP_PORT = 1935
WEB_PORT = 5000

DRONE_URLS = {
    "drone1": "rtsp://127.0.0.1:8554/live/drone1",
    "drone2": "rtsp://127.0.0.1:8554/live/drone2",
}

MEDIAMTX_CONFIG = f"""
logLevel: info
rtmp: yes
rtmpAddress: :{RTMP_PORT}
rtsp: yes
paths:
  "~^.*$":
    source: publisher
""".lstrip()

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU Found: {gpu_name}")
    DEVICE = "cuda:0"
else:
    logger.warning("GPU not found. Running on CPU.")
    DEVICE = "cpu"

GPU_LOCK = threading.Lock()
if os.path.exists("best.pt"):
    logger.info("Loading Custom Airsoft Model (best.pt)...")
    model = YOLO("best.pt").to(DEVICE)
else:
    logger.warning("Custom model 'best.pt' not found! Using standard YOLOv8n.")
    model = YOLO("yolov8n.pt").to(DEVICE)


class VideoStreamer:
    def __init__(self, drone_id: str, rtsp_url: str):
        self.drone_id = drone_id
        self.rtsp_url = rtsp_url
        self.running = False

        self.lock = threading.Lock()
        self.raw_frame = None
        self.jpeg_bytes = None

        self.detect_interval = 0.1
        self.last_ai_time = 0.0
        self.detections = []

    def start(self):
        if self.running: return
        self.running = True
        logger.info(f"Starting {self.drone_id} workers...")
        threading.Thread(target=self._capture_worker, daemon=True).start()
        threading.Thread(target=self._process_worker, daemon=True).start()

    def stop(self):
        self.running = False

    def _capture_worker(self):
        while self.running:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                time.sleep(1.0)
                continue

            logger.info(f" {self.drone_id} CONNECTED via RTSP")

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f" {self.drone_id} signal lost")
                    break

                with self.lock:
                    self.raw_frame = frame

            cap.release()
            time.sleep(1.0)

    def _process_worker(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.raw_frame is not None:
                    frame_to_process = self.raw_frame.copy()

            if frame_to_process is None:
                time.sleep(0.02)
                continue

            now = time.time()
            if now - self.last_ai_time > self.detect_interval:
                try:
                    with GPU_LOCK:
                        results = model.track(
                            frame_to_process,
                            imgsz=640,
                            conf=0.5,
                            classes=[0],
                            persist=True,
                            tracker="bytetrack.yaml",
                            verbose=False
                        )
                    self.detections = results[0].boxes
                    self.last_ai_time = now
                except Exception as e:
                    logger.error(f"AI Error: {e}")

            # Drawing Logic
            if self.detections is not None:
                for box in self.detections:

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    track_id = int(box.id[0]) if box.id is not None else 0

                    label = f"TARGET #{track_id}" if track_id > 0 else "TARGET"

                    cv2.rectangle(frame_to_process, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame_to_process, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

                    cv2.putText(frame_to_process, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            frame_resized = cv2.resize(frame_to_process, (854, 480))
            _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            with self.lock:
                self.jpeg_bytes = buffer.tobytes()

            time.sleep(0.01)

    def generate(self):
        while self.running:
            with self.lock:
                jpg = self.jpeg_bytes

            if jpg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

            time.sleep(0.04)


streamers: Dict[str, VideoStreamer] = {}
mediamtx_proc = None


def start_mediamtx():
    global mediamtx_proc
    CFG_PATH.write_text(MEDIAMTX_CONFIG)

    if not MEDIAMTX_BIN.exists():
        logger.warning(f"MediaMTX binary not found at {MEDIAMTX_BIN}. Skipping server start.")
        return

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
    time.sleep(2)


def stop_mediamtx():
    global mediamtx_proc
    if mediamtx_proc:
        logger.info("Stopping MediaMTX...")
        mediamtx_proc.terminate()
        try:
            mediamtx_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            mediamtx_proc.kill()
        mediamtx_proc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== STARTUP ===")
    start_mediamtx()

    for name, url in DRONE_URLS.items():
        s = VideoStreamer(name, url)
        streamers[name] = s
        s.start()

    yield

    logger.info("=== SHUTDOWN ===")
    for s in streamers.values():
        s.stop()
    stop_mediamtx()


app = FastAPI(lifespan=lifespan)


@app.get("/detect/{drone_id}")
def detect_feed(drone_id: str):
    if drone_id not in streamers:
        return HTMLResponse("Drone not found", status_code=404)

    return StreamingResponse(
        streamers[drone_id].generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def index():
    grid_html = "".join(
        f"""
        <div class="cell">
            <h3>{name.upper()}</h3>
            <div class="vid-wrapper">
                <img src="/detect/{name}" />
            </div>
        </div>
        """
        for name in DRONE_URLS
    )

    return f"""
    <!doctype html>
    <html>
    <head>
        <title>Drone Surveillance</title>
        <style>
            body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }}
            h2 {{ border-bottom: 1px solid #444; padding-bottom: 10px; }}
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .cell {{ flex: 1 1 400px; background: #2a2a2a; padding: 10px; border-radius: 8px; }}
            .vid-wrapper {{ position: relative; width: 100%; padding-top: 56.25%; background: #000; }}
            img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }}
        </style>
    </head>
    <body>
        <h2>Drone Surveillance</h2>
        <p>Active Device: {DEVICE}</p>
        <div class="grid">
            {grid_html}
        </div>
    </body>
    </html>
    """


def main():
    print(f"\nStream Ingest: rtmp://127.0.0.1:{RTMP_PORT}/live/<drone_id>")
    print(f"Web Interface: http://127.0.0.1:{WEB_PORT}\n")

    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, access_log=False)


if __name__ == "__main__":
    main()