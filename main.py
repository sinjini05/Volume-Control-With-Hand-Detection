import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Function to handle errors
def handle_error(e):
    print(f"An error occurred: {e}")

# Function to initialize camera with frame width and height
def initialize_camera(frame_width=640, frame_height=480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    if not cap.isOpened():
        raise Exception("Could not open video device.")
    return cap

# Function to set volume based on distance
def set_volume(length, volMin, volMax, volume_control):
    volume = max(volMin, min(volMax, int((length / 200) * (volMax - volMin) + volMin)))
    volume_control.SetMasterVolumeLevel(volume, None)

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # Initialize the hand detection
mpDraw = mp.solutions.drawing_utils  # Drawing utility for hand landmarks

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_control = cast(interface, POINTER(IAudioEndpointVolume))

# Initialize volume limits
volMin = -65.25  # Minimum volume in dB
volMax = 0.0     # Maximum volume in dB

# Initialize video capture with a smaller frame size
try:
    cap = initialize_camera(frame_width=320, frame_height=240)  # Resize to 320x240
except Exception as e:
    handle_error(e)
    exit(1)

# Use a try block for the main loop
try:
    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img = cv2.resize(img, (320, 240))  # Resize if necessary
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
            set_volume(length, volMin, volMax, volume_control)  # Call the set_volume function

        # Display the current volume level in dB
        current_volume = volume_control.GetMasterVolumeLevel()
        cv2.putText(img, f'Volume: {int(current_volume)} dB', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Image', img)

        if cv2.waitKey(33) & 0xFF == ord('q'):  # Exit on 'q'
            break

except cv2.error as e:
    handle_error(e)
except Exception as e:
    handle_error(e)

finally:
    cap.release()
    cv2.destroyAllWindows()
