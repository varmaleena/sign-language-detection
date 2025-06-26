from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('model/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        H, W, _ = frame.shape

        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        results = hands.process(frame_rgb)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb, 
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                    mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                prediction = model.predict([np.array(data_aux)[:42]])[0]

                cv2.rectangle(frame_rgb, (x1, y1 - 10), (x2, y2), (255, 99, 173), 6)
                cv2.putText(frame_rgb, prediction, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame_rgb)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
