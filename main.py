# Function to initialize camera with frame width and height
def initialize_camera(frame_width=640, frame_height=480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    if not cap.isOpened():
        raise Exception("Could not open video device.")
    return cap

# Initialize video capture with a smaller frame size
try:
    cap = initialize_camera(frame_width=320, frame_height=240)  # Resize to 320x240
except Exception as e:
    handle_error(e)
    exit(1)

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Resize the image (if needed) - optional since we set it earlier
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
            set_volume(length, volMin, volMax)

        cv2.putText(img, f'Volume: {int(volume.GetMasterVolumeLevel())}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Image', img)
        
        # Add delay for lower frame rate (e.g., 30 FPS)
        if cv2.waitKey(33) & 0xFF == ord('q'):  # 1000 ms / 30 FPS ≈ 33 ms
            break

    except cv2.error as e:
        handle_error(e)
    except Exception as e:
        handle_error(e)

# Release resources
finally:
    cap.release()
    cv2.destroyAllWindows()
