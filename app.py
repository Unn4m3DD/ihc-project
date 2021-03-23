import socket
import struct
import cv2
import mediapipe as mp
import signal
import sys
from math import atan2, degrees, radians, sqrt
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

OFFSET = 0.1

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def signal_handler(signal, frame):
  sock.close()
  sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def distance_2d(p1, p2):
  return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def process_hand(hand_landmarks, image_width):
  inside_bounds = is_inside_offset(hand_landmarks)
  index_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
  middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
  middle_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
  pinky_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
  ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
  wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

  angle_x = -degrees(atan2(
      pinky_finger_insertion.z - index_finger_insertion.z,
      distance_2d(pinky_finger_insertion, index_finger_insertion)
  ))

  angle_y = degrees(atan2(
      (middle_finger_insertion.y - wrist.y),
      (middle_finger_insertion.x - wrist.x)
  ))

  angle_z = -degrees(atan2(
      wrist.z - middle_finger_tip.z,
      distance_2d(wrist, middle_finger_tip)
  ))

  closed = (sqrt(
      (ring_finger_tip.x - wrist.x) ** 2 +
      (ring_finger_tip.y - wrist.y) ** 2 +
      (ring_finger_tip.z - wrist.z) ** 2
  ) / sqrt(
      (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 +
      (pinky_finger_insertion.y - index_finger_insertion.y) ** 2 +
      (pinky_finger_insertion.z - index_finger_insertion.z) ** 2
  )) / 2.5 - .45

  if(closed > 1):
    closed = 1
  if(closed < 0):
    closed = 0

  is_right_hand = wrist.x > image_width / 2
  print(" "*50, end="\r")
  print(
      f"~angle_x: {angle_x:6.2f}º, angle_y: {angle_y:6.2f}º, closed metric {closed:.10f}, {inside_bounds}", end="\r")
  sock.sendall(struct.pack('ldddd', is_right_hand,
               angle_x, angle_y, angle_z, closed))


def get_hand_info():
  sock.connect(('127.0.0.1', 42069))
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
          min_detection_confidence=0.8,
          min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
      success, image = cap.read()
      height, width, _ = image.shape
      acceptable_space = ((int(width*OFFSET), int(height*OFFSET)),
                          (int(width-width*OFFSET), int(height-height*OFFSET)))
      if not success:
        print("Ignoring empty camera frame.")
        continue
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      results = hands.process(image)
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          process_hand(hand_landmarks, width)
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      cv2.rectangle(image, acceptable_space[0], acceptable_space[1],
                    color=(0, 255, 0), thickness=2)
      cv2.imshow('IHC Hand Recognition', image)

      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
  sock.close()


def is_inside_offset(hand_landmarks):
  for i in range(len(list(map(int, mp_hands.HandLandmark)))):
    if not (hand_landmarks.landmark[i].x > OFFSET and hand_landmarks.landmark[i].x < 1-OFFSET and hand_landmarks.landmark[i].y > OFFSET and hand_landmarks.landmark[i].y < 1-OFFSET):
      return False
  return True


#  #   ###
# ###  ###
#  #   ###

if __name__ == "__main__":
  get_hand_info()
