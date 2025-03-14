import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import flask
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import time
import json

app = Flask(__name__)

# Load model
model = load_model('model_1.h5', compile=False)
class_labels = ['Mengantuk & Menguap', 'Mengantuk & Tidak Menguap', 'Menguap & Tidak Mengantuk']

# Camera control
camera = None
is_running = False
lock = threading.Lock()
latest_prediction = {"class": "", "confidence": 0.0, "status": "inactive"}
prediction_lock = threading.Lock()

def process_frame(frame):
    try:
        # Convert BGR to RGB and resize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (64, 64))
        normalized = resized.astype('float32') / 255.0
        
        # Prediction
        input_frame = np.expand_dims(normalized, axis=0)
        prediction = model.predict(input_frame, verbose=0)
        
        # Update prediction data
        class_index = np.argmax(prediction)
        with prediction_lock:
            latest_prediction.update({
                "class": class_labels[class_index],
                "confidence": float(prediction[0][class_index]),
                "timestamp": time.time()
            })
            
        return frame
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return frame

def generate_frames():
    global camera
    with lock:
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Camera not available")
                
            while is_running:
                success, frame = camera.read()
                if not success:
                    break
                    
                processed_frame = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)
                
        except Exception as e:
            print(f"Camera error: {str(e)}")
        finally:
            if camera:
                camera.release()
            camera = None

def generate_predictions():
    while True:
        with prediction_lock:
            data = json.dumps(latest_prediction)
        yield f"data: {data}\n\n"
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction_updates')
def prediction_updates():
    return Response(generate_predictions(), mimetype='text/event-stream')

@app.route('/start', methods=['POST'])
def start():
    global is_running
    if not is_running:
        is_running = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route('/system_status')
def system_status():
    return jsonify({
        "camera": "connected" if camera and camera.isOpened() else "disconnected",
        "model": "loaded",
        "resolution": "640x480"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)