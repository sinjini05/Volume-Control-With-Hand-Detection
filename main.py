import cv2 
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np 

# Function to initialize camera
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device.")
    return cap

# Function to set volume with validation
def set_volume(length, volMin, volMax):
    if 15 < length < 220:
        vol = np.interp(length, [15, 220], [volMin, volMax])
        volume.SetMasterVolumeLevel(vol, None)
        print(f"Volume set to: {vol}, Length: {length}")
    else:
        print(f"Length out of range: {length}")

# Function to handle errors and log them
def handle_error(e):
    print(f"An error occurred: {e}")

# Initialize video capture
try:
    cap = initialize_camera()
except Exception as e:
    handle_error(e)
    exit(1)

# Initialize MediaPipe hands
mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize audio volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []
        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy]) 
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)
        
        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = hypot(x2 - x1, y2 - y1)
            set_volume(length, volMin, volMax)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error as e:
        handle_error(e)
    except Exception as e:
        handle_error(e)

# Release resources
finally:
    cap.release()
    cv2.destroyAllWindows()
