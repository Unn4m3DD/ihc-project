import socket
import cv2
import mediapipe as mp
from math import atan2, degrees, radians, sqrt;
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

OFFSET = 0.1

# For webcam input:
def get_hand_info():
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect(('localhost', 42069))
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
      min_detection_confidence=0.8,
      min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
      success, image = cap.read()
      height, width, _ = image.shape
      acceptable_space = ((int(width*OFFSET), int(height*OFFSET)), (int(width-width*OFFSET), int(height-height*OFFSET)))
      status = "outside"
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

      angle_x = angle_y = closed = 0
      direction = [0,0]
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          if is_inside_offset(hand_landmarks): status = "inside" 


          index_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
          middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
          pinky_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
          ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
          wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
          middle_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]


          angle_x = degrees(atan2(
            (pinky_finger_insertion.z - index_finger_insertion.z), 
            (
              sqrt(
                (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 + 
                (pinky_finger_insertion.y - index_finger_insertion.y) ** 2
              )
            )
          ))

          angle_y = degrees(atan2(
            (wrist.z - middle_finger_insertion.z), 
            (
              sqrt(
                (wrist.x - middle_finger_insertion.x) ** 2 + 
                (wrist.y - middle_finger_insertion.y) ** 2
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
          print(f"~angle_x: {angle_x:6.2f}º, angle_y: {angle_y:6.2f}º, closed metric {closed:.10f}, direction ({direction[0]:.10f}, {direction[1]:.10f}), {status}", end="\r")
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      
      s.sendall('{0}{1}{2}{3}{4}'.format(angle_x, angle_y, closed, direction[0], direction[1]))
      cv2.rectangle(image, acceptable_space[0], acceptable_space[1], color=(0,255,0) if status == "inside" else (0,0,255), thickness=2)
      cv2.imshow('IHC Hand Recognition', image)
      
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
  s.close()

def is_inside_offset(hand_landmarks):
  for i in range(len(list(map(int, mp_hands.HandLandmark)))):
    if not (hand_landmarks.landmark[i].x > OFFSET and hand_landmarks.landmark[i].x < 1-OFFSET and hand_landmarks.landmark[i].y > OFFSET and hand_landmarks.landmark[i].y < 1-OFFSET): return False
  return True


#  #   ###
# ###  ###
#  #   ###

if __name__ == "__main__":
  get_hand_info()