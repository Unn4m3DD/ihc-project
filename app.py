import socket
import struct
import cv2
import mediapipe as mp
import signal
import sys
import time
from math import atan2, degrees, radians, sqrt
import models.hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SOCKET_CONNECTION = True
HOST = '127.0.0.1'
PORT = 42069

# camera
CAMERA_OFFSET = 0.1
NUM_HANDS = 1

def signal_handler(signal, frame):
  sock.close()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_hand_info():
  if SOCKET_CONNECTION:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      sock.connect((HOST, PORT))
    except Exception as e:
      print(str(e))
      time.sleep(4)
      return
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
          max_num_hands=NUM_HANDS,
          min_detection_confidence=0.8,
          min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
      success, image = cap.read()
      height, width, _ = image.shape
      acceptable_space = ((int(width*CAMERA_OFFSET), int(height*CAMERA_OFFSET)),
                          (int(width-width*CAMERA_OFFSET), int(height-height*CAMERA_OFFSET)))
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

      '''
      if NUM_HANDS == 2:
        if results.multi_handedness:
          for hand_landmarks in results.multi_hand_landmarks:
            pass
      '''


      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          if is_inside_offset(hand_landmarks):
            status = "inside"

          index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
          index_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
          middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
          middle_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
          pinky_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
          ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
          ring_finger_insertion = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
          wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

          angle_z = -degrees(atan2(
              (wrist.z - middle_finger_tip.z),
              (
                  sqrt(
                      (wrist.x - middle_finger_tip.x) ** 2 +
                      (wrist.y - middle_finger_tip.y) ** 2
                  )
              )
          ))

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
              (middle_finger_insertion.y - wrist.y),
              (middle_finger_insertion.x - wrist.x)
          ))

          '''
          closed = (((sqrt(
              (ring_finger_tip.x - wrist.x) ** 2 +
              (ring_finger_tip.y - wrist.y) ** 2 +
              (ring_finger_tip.z - wrist.z) ** 2
          ) + sqrt(
              (ring_finger_tip.x - wrist.x) ** 2 +
              (ring_finger_tip.y - wrist.y) ** 2 +
              (ring_finger_tip.z - wrist.z) ** 2
          ))/2) / sqrt(
              (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 +
              (pinky_finger_insertion.y - index_finger_insertion.y) ** 2 +
              (pinky_finger_insertion.z - index_finger_insertion.z) ** 2 * 0.4
          ) ) / 2.5 - .45
          '''

          closed = (sqrt(
              (ring_finger_tip.x - wrist.x) ** 2 +
              (ring_finger_tip.y - wrist.y) ** 2 +
              (ring_finger_tip.z - wrist.z) ** 2
          ) / sqrt(
              (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 +
              (pinky_finger_insertion.y - index_finger_insertion.y) ** 2 +
              (pinky_finger_insertion.z - index_finger_insertion.z) ** 2 * 0.5
          ) ) / 2.5 - .45
          
          if(closed > 1):
            closed = 1
          if(closed < 0):
            closed = 0
          


          print(f"                                         ", end="\r")
          print(
              f"~angle_x: {angle_x:6.2f}º, angle_y: {angle_y:6.2f}º, closed metric {closed:.10f}, {status}", end="\r")
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      if SOCKET_CONNECTION and status == 'inside':
        try:
          sock.sendall(struct.pack('dddd', angle_x, angle_y, angle_z, closed))
        except Exception as e:
          print(str(e))
          sock.close()
          return # block before rendering image on screen in order for to user to see that there is a problem with internet

      cv2.rectangle(image, acceptable_space[0], acceptable_space[1], color=(
          0, 255, 0) if status == "inside" else (0, 0, 255), thickness=2)
      cv2.imshow('IHC Hand Recognition', image)

      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
  sock.close()

def is_inside_offset(hand_landmarks):
  for i in range(len(list(map(int, mp_hands.HandLandmark)))):
    if not (hand_landmarks.landmark[i].x > CAMERA_OFFSET and hand_landmarks.landmark[i].x < 1-CAMERA_OFFSET and hand_landmarks.landmark[i].y > CAMERA_OFFSET and hand_landmarks.landmark[i].y < 1-CAMERA_OFFSET):
      return False
  return True

if __name__ == "__main__":
  if NUM_HANDS <= 2 and NUM_HANDS > 0:
    while(True):
      get_hand_info()
  else:
    print("number of hands not supported")

  
'''
Interface




'''