import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
pg.FAILSAFE = True
pg.PAUSE = 0.5
import pygetwindow as pgw
import time
import math

def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) ** .5

active = pgw.getActiveWindow()
handsDetector = mp.solutions.hands.Hands(static_image_mode=False,
                   max_num_hands=1,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.9)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)
prev_fist = False   
sensitivity = 120
x = 0
y = 0
x1 = x
y1 = y
prev_fist=False
while(cap.isOpened()):
    active = pgw.getActiveWindow()
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    results = handsDetector.process(flippedRGB)
    results_face = face_mesh.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        if results_face.multi_face_landmarks is not None:
            for face_landmarks in results_face.multi_face_landmarks:
                nose = face_landmarks.landmark[0]
        x1, y1 = x, y
        time.sleep(0.075)
        cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0, (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        bf = results.multi_hand_landmarks[0].landmark[12]
        sf = results.multi_hand_landmarks[0].landmark[4]
        pf = results.multi_hand_landmarks[0].landmark[8]
        if "Left" in str(results.multi_handedness[0]):
            hand = "Left"
        else:
            hand = "Right"
        if 2 * r / ws > 1.35:
                cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 0, 255), 2)
                prev_fist = False
                if int(x)-int(x1)<=-sensitivity and hand == 'Right' and abs(int(x)-int(x1))>=sensitivity:
                    pg.hotkey('alt', 'esc')
                    pgw.getActiveWindow().maximize()
                    time.sleep(0.5)
                elif math.hypot((bf.x - sf.x) * flippedRGB.shape[1], (bf.y - sf.y) * flippedRGB.shape[0]) <= 40:
                    pg.hotkey('win', 'shift', 's')
                elif math.hypot((nose.x - pf.x) * flippedRGB.shape[1], (nose.y - pf.y) * flippedRGB.shape[0]) <= 45:
                    pg.press('volumemute') 
        else:
            cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 255, 0), 2)
            if not prev_fist:
                if active is not None:
                    active.minimize()
                    prev_fist = True
                    time.sleep(0.5)
handsDetector.close()
face_mesh.close()
