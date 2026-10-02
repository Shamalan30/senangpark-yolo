import cv2
import json
import time
import threading
from flask import Flask, jsonify, Response
from flask_cors import CORS
from detector import detect_parking, get_summary

app = Flask(__name__)
CORS(app)

latest_status = []
latest_summary = {}
latest_frame = None
latest_jpeg = None  # Pre-encoded JPEG bytes
lock = threading.Lock()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag

def camera_loop():
    global latest_status, latest_summary, latest_frame, latest_jpeg
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1

        # Run YOLO every 3 frames to reduce CPU load
        if frame_count % 3 == 0:
            zone_status, annotated_frame = detect_parking(frame)
            summary = get_summary(zone_status)

            # Pre-encode JPEG for faster response
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)

            with lock:
                latest_status = zone_status
                latest_summary = summary
                latest_frame = annotated_frame.copy()
                latest_jpeg = buffer.tobytes()
        else:
            # For non-detection frames just encode raw
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            with lock:
                if latest_jpeg is None:
                    latest_jpeg = buffer.tobytes()

thread = threading.Thread(target=camera_loop, daemon=True)
thread.start()

@app.route("/api/parking/status")
def parking_status():
    with lock:
        return jsonify({
            "success": True,
            "zones": latest_status,
            "summary": latest_summary,
            "timestamp": time.time()
        })

@app.route("/api/parking/available")
def available_slots():
    with lock:
        available = [z for z in latest_status
                     if z["status"] == "available"]
        return jsonify({
            "success": True,
            "available_slots": available,
            "count": len(available)
        })

@app.route("/api/parking/summary")
def parking_summary():
    with lock:
        return jsonify(latest_summary)

@app.route("/snapshot")
def snapshot():
    """Returns pre-encoded JPEG — very fast!"""
    with lock:
        if latest_jpeg is None:
            return "No frame", 404
        jpeg = latest_jpeg
    return Response(jpeg, mimetype="image/jpeg")

@app.route("/video_feed")
def video_feed():
    """MJPEG stream for browser viewing."""
    def generate():
        while True:
            with lock:
                if latest_jpeg is None:
                    time.sleep(0.05)
                    continue
                frame_bytes = latest_jpeg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame_bytes + b"\r\n")
            time.sleep(0.033)  # ~30fps
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)