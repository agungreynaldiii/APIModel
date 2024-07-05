import argparse
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ultralytics import YOLO
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

# Read CSV file once when the server starts
food_info = pd.read_csv('food.csv')

def get_food_info(label):
    info = food_info[food_info['Makanan'].str.lower() == label.lower()]
    if not info.empty:
        return info.to_dict(orient='records')[0]
    else:
        return None

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        try:
            if 'file' in request.files:
                f = request.files['file']
                basepath = os.path.dirname(__file__)
                filepath = os.path.join(basepath, UPLOAD_FOLDER, f.filename) # type: ignore
                f.save(filepath)
                
                file_extension = f.filename.rsplit('.', 1)[1].lower() # type: ignore
                
                model = YOLO('best.pt')

                if file_extension in ['jpg', 'jpeg', 'png']:
                    img = cv2.imread(filepath)
                    detections = model(img, save=True)

                    # Extract the class names from the detections
                    labels = [model.names[int(cls)] for cls in detections[0].boxes.cls] # type: ignore

                    # Get additional info from CSV
                    additional_info = []
                    for label in labels:
                        info = get_food_info(label)
                        if info:
                            additional_info.append(info)

                    # Save the detected image with a generic name
                    detected_img_path = os.path.join(DETECT_FOLDER, 'image0.jpg')
                    cv2.imwrite(detected_img_path, detections[0].plot())

                    # Full URL to the detected image
                    full_detected_img_path = f"http://{request.host}/detect/image0.jpg"

                    return jsonify({'labels': labels, 'filename': f.filename, 'detected_img_path': full_detected_img_path, 'additional_info': additional_info})
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
            
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