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

                # Normalize the angle to keep it consistent across all hand orientations
                if angle_deg < 0:
                    angle_deg += 360

                # Scaling factor based on a more stable bone length (wrist to knuckle)
                scale_factor = distance / 500.0  # Adjust scaling divisor as needed

                # Correct angle to ensure the bracelet is oriented properly
                if wrist_x < index_x:
                    angle_deg -= 110
                else:
                    angle_deg += 110

                angle_deg = -angle_deg

                # Overlay the bracelet at the wrist
                frame = overlay_bracelet(frame, bracelet, wrist_x, wrist_y, angle_deg, scale_factor)

            except Exception as e:
                print(f"Error processing hand landmarks: {e}")