import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.to("cuda")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opened!")
    exit()

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
  <head>
    <title>Drone Feed + YOLO Detection</title>
  </head>
  <body style="margin:0;background:black">
    <img src="{{ url_for('video_feed') }}" style="width:100%;height:auto;">
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(PAGE)

def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        results = model(frame, conf=0.5, classes=[0])
        annotated_frame = results[0].plot()
        ok, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("YOLO human detection server started!")
    print("Open http://127.0.0.1:5000 or http://<LAN_IP>:5000 on your phone")
    app.run(host='0.0.0.0', port=5000)