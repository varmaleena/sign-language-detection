#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)


# In[ ]:

def run_sign_rec(frame):

    data_dir = 'asl_dataset'
    data = []
    labels = []

    for i in sorted(os.listdir(data_dir)):
        if i == '.DS_Store':
            continue
        else:
            dir_path = os.path.join(data_dir, i)
            for j in os.listdir(dir_path):
                if j == '.DS_Store':
                    continue
                
                img_path = os.path.join(dir_path, j)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux = []
                        for z in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[z].x
                            y = hand_landmarks.landmark[z].y
                            data_aux.append(x)
                            data_aux.append(y)
                        data.append(data_aux)
                        labels.append(i)
                    
    # Save data
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)


    # In[ ]:


    data_dir = 'asl_dataset'
    for i in sorted(os.listdir(data_dir)):
        if i == '.DS_Store':
            pass
        else:
            for j in os.listdir(os.path.join(data_dir, i))[0:1]:
                img_path = os.path.join(data_dir, i, j)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img_rgb,  # img to draw
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                plt.figure()
                plt.title(i)
                plt.imshow(img_rgb)
                plt.show()
                plt.close() 


    # In[ ]:


    X_train, X_test, y_train, y_test = train_test_split(np.array(data), labels, test_size=0.15, random_state=22, shuffle=True)

    # model
    model = RandomForestClassifier(random_state=22)
    model.fit(X_train,y_train)

    # predict
    pred=model.predict(X_test)

    # accruracy
    accuracy_score(y_test,pred)


    # In[ ]:


    f = open('model.p', 'wb')
    pickle.dump({'model':model},f)
    f.close() 


    # In[ ]:


    model_dict = pickle.load(open('model.p','rb'))
    model = model_dict['model']


    # In[ ]:


    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():

            data_aux=[]
            x_ = []
            y_ = []

            ret, frame = cap.read()
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True 
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb, # img to draw
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                        mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                    )


                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

                x1 = int(min(x_) * W)-10
                y1 = int(min(y_) * H)-10

                x2 = int(max(x_) * W)-10
                y2 = int(max(y_) * H)-10
                prediction = model.predict([np.array(data_aux)[0:42]])[0]

                cv2.rectangle(frame_rgb, (x1,y1-10), (x2,y2), (255,99,173), 6)
                cv2.putText(frame_rgb, prediction, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 5, (255,0,0), 5, cv2.LINE_AA)

            cv2.imshow('frame',frame_rgb)  
            # cv2.waitKey(1)q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


    # In[ ]:





    # In[ ]:





    # In[ ]:




