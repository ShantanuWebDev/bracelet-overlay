    for hand_landmarks in results.multi_hand_landmarks:
            try:
                # Get wrist landmark (Landmark 0 is the wrist in Mediapipe Hand Model)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                h, w, c = frame.shape
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                pinky_x, pinky_y = int(pinky.x * w), int(pinky.y * h)

                # Calculate the distance between wrist and pinky to determine scale factor
                distance = calculate_distance((wrist_x, wrist_y), (pinky_x, pinky_y))

                # Calculate the angle between wrist and pinky
                delta_x = pinky_x - wrist_x
                delta_y = pinky_y - wrist_y
                angle_rad = np.arctan2(delta_y, delta_x)  # Angle in radians
                angle_deg = np.degrees(angle_rad)  # Convert to degrees

                # Debug: Check the distance and scale factor
                print(f"Distance between wrist and pinky: {distance}")
                scale_factor = distance / 850.0  # Adjusted scaling factor
                print(f"Scale Factor: {scale_factor}")
                print(f"Initial Angle (deg): {angle_deg}")

                # Adjust angle based on hand orientation
                if wrist_x < pinky_x:  # Right hand
                    angle_deg -= 90  # Adjust for right hand
                else:  # Left hand
                    angle_deg += 90  # Adjust for left hand

                # Flip angle if needed
                angle_deg = -angle_deg  # Flip if required

                # Overlay the bracelet at wrist coordinates
                frame = overlay_bracelet(frame, bracelet, wrist_x, wrist_y, angle_deg, scale_factor)

                # Optional: Draw hand landmarks for debugging
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            except Exception as e:
                print(f"Error processing hand landmarks: {e}")