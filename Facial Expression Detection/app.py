from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# Load the emotion detection model
model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# OpenCV video capture (camera)
cap = cv2.VideoCapture(0)

# Store last detected emotion
global_last_emotion = "Neutral"

def generate():
    global global_last_emotion
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)  # shape: (1, 48, 48, 1)

            # Predict emotion
            prediction = model.predict(roi)
            label = emotion_labels[np.argmax(prediction)]

            # Update last detected emotion
            global_last_emotion = label

            # Draw rectangle around face and add emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode the frame in JPEG format and return as a byte stream
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Home page with webcam
@app.route('/')
def index():
    return render_template('index.html')

# Streaming video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Emotion feed route (for chart updating)
@app.route('/emotion_feed')
def emotion_feed():
    global global_last_emotion
    return jsonify({"emotion": global_last_emotion, "timestamp": time.time()})

if __name__ == '__main__':
    app.run(debug=True)
