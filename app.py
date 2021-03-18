import cv2
import mediapipe as mp
from math import atan2, degrees, radians, sqrt;
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        index_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        pinky_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        angle = degrees(atan2(
          (pinky_finger_insertion.z - index_finger_insertion.z), 
          (
            sqrt(
              (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 + 
              (pinky_finger_insertion.y - index_finger_insertion.y) ** 2
            )
          )
        ))

        closed = sqrt(
              (ring_finger_tip.x - wrist.x) ** 2 + 
              (ring_finger_tip.y - wrist.y) ** 2 + 
              (ring_finger_tip.z - wrist.z) ** 2
            ) / sqrt(
              (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 + 
              (pinky_finger_insertion.y - index_finger_insertion.y) ** 2
            )
        
        direction = (
          middle_finger_tip.x - wrist.x,
          middle_finger_tip.y - wrist.y
        )
        print(f"                                         ", end="\r")
        print(f"~{angle:.2f}º, closed metric {closed}, direction ({direction[0]}, {direction[1]})", end="\r")
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


#  #   ###
# ###  ###
#  #   ###