import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

bracelet = cv2.imread('brc4.png', cv2.IMREAD_UNCHANGED)

if bracelet is None:
    print("Bracelet image could not be loaded. Check file path.")
else:
    print(f"Bracelet image loaded with shape: {bracelet.shape}")

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def overlay_bracelet(frame, bracelet, wrist_x, wrist_y, angle_deg, scale_factor):
    bracelet_h, bracelet_w, _ = bracelet.shape
    scale_factor = min(scale_factor, 0.5)
    new_bracelet_w = int(bracelet_w * scale_factor)
    new_bracelet_h = int(bracelet_h * scale_factor)

    if new_bracelet_w < 20 or new_bracelet_h < 20:
        new_bracelet_w = 20
        new_bracelet_h = 20

    resized_bracelet = cv2.resize(bracelet, (new_bracelet_w, new_bracelet_h))
    center = (new_bracelet_w // 2, new_bracelet_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((new_bracelet_h * sin) + (new_bracelet_w * cos))
    new_h = int((new_bracelet_h * cos) + (new_bracelet_w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_bracelet = cv2.warpAffine(resized_bracelet, rotation_matrix, (new_w, new_h))

    x_offset = int(wrist_x - new_w / 2)
    y_offset = int(wrist_y - new_h / 2 + 50)

    x_offset = max(0, min(x_offset, frame.shape[1] - new_w))
    y_offset = max(0, min(y_offset, frame.shape[0] - new_h))

    for c in range(0, 3):
        frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c] = \
            rotated_bracelet[:, :, c] * (rotated_bracelet[:, :, 3] / 255.0) + \
            frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c] * \
            (1.0 - rotated_bracelet[:, :, 3] / 255.0)

    return frame

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),  # Customize landmark dots
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Customize connections between landmarks
            )
            try:
                # Get wrist and index finger's knuckle (MCP joint)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                h, w, c = frame.shape
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                index_x, index_y = int(index_knuckle.x * w), int(index_knuckle.y * h)

                # Calculate distance between wrist and index knuckle for scaling
                distance = calculate_distance((wrist_x, wrist_y), (index_x, index_y))

                delta_x = index_x - wrist_x
                delta_y = index_y - wrist_y
                angle_rad = np.arctan2(delta_y, delta_x)
                angle_deg = np.degrees(angle_rad)

                # Keep angle horizontal relative to hand tilt
                angle_deg = 110 - angle_deg                     
                # Scaling factor based on a more stable bone length (wrist to knuckle)
                scale_factor = distance / 500.0  # Adjust scaling divisor as needed

                # Overlay the bracelet at the wrist
                frame = overlay_bracelet(frame, bracelet, wrist_x, wrist_y, angle_deg, scale_factor)

            except Exception as e:
                print(f"Error processing hand landmarks: {e}")


    cv2.imshow('Bracelet Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
