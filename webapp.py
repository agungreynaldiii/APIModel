import argparse
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, UPLOAD_FOLDER, f.filename)
            f.save(filepath)
            
            file_extension = f.filename.rsplit('.', 1)[1].lower()
            
            model = YOLO('best.pt')

            if file_extension in ['jpg', 'jpeg', 'png']:
                img = cv2.imread(filepath)
                detections = model(img, save=True)

                # Extract the class names from the detections
                labels = [model.names[int(cls)] for cls in detections[0].boxes.cls]

                # Save the detected image with a generic name
                detected_img_path = os.path.join(DETECT_FOLDER, 'image0.jpg')
                cv2.imwrite(detected_img_path, detections[0].plot())

                return jsonify({'labels': labels, 'filename': f.filename, 'detected_img_path': detected_img_path})
            
    return render_template('index.html')

@app.route('/uploads/<filename>')
def display_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect/<filename>')
def display_detect(filename):
    return send_from_directory(DETECT_FOLDER, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)