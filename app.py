import socket
import struct
import cv2
import mediapipe as mp
import signal
import sys
import time
from math import atan2, degrees, radians, sqrt
from models.hands import Hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SOCKET_CONNECTION = False
HOST = '127.0.0.1'
PORT = 42069

# camera
CAMERA_OFFSET = 0.1
NUM_HANDS = 2


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

      hands_list = []
      if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
          if is_inside_offset(hand_landmarks):
            status = "inside"
          hand = Hand(results.multi_handedness[i].classification[0])
          hand.hand_calculations(hand_landmarks)
          hands_list.append(hand)
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

          # show hand informations
          wrist = hand_landmarks.landmark[hand.mp_hands.HandLandmark.WRIST]
          cv2.putText(image, hand.classification.label[0], (int(wrist.x*width-10), int(
              wrist.y*height-40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
          print(f"                                         ", end="\r")
          print(f"~angle_x: {hand.angle_x:6.2f}º, angle_y: {hand.angle_y:6.2f}º, closed metric {hand.closed:.10f}, mao: {hand.classification.label},{status}", end="\r")

      if SOCKET_CONNECTION and status == 'inside':
        try:
          if NUM_HANDS == 1:
            sock.sendall(struct.pack(
                'dddd', hands_list[0].angle_x, hands_list[0].angle_y, hands_list[0].angle_z, hands_list[0].closed))
          elif NUM_HANDS == 2:
            sock.sendall(struct.pack(
                'dddd', hands_list[0].angle_x, hands_list[0].angle_y, hands_list[0].angle_z, hands_list[0].closed))  # just for now
            # sock.sendall(struct.pack('bddddbdddd', angle_x, angle_y, angle_z, closed))
            pass
        except Exception as e:
          print(str(e))
          sock.close()
          return  # block before rendering image on screen in order for to user to see that there is a problem with internet
      cv2.rectangle(image, acceptable_space[0], acceptable_space[1], color=(
          0, 255, 0) if status == "inside" else (0, 0, 255), thickness=2)
      cv2.imshow('IHC Hand Recognition', image)

      if cv2.waitKey(5) & 0xFF == 27:
        break

    cap.release()
    sock.close()


def is_inside_offset(hand_landmarks):
  for i in list(map(int, mp_hands.HandLandmark)):
    if not (hand_landmarks.landmark[i].x > CAMERA_OFFSET and hand_landmarks.landmark[i].x < 1-CAMERA_OFFSET and hand_landmarks.landmark[i].y > CAMERA_OFFSET and hand_landmarks.landmark[i].y < 1-CAMERA_OFFSET):
      return False
  return True


if __name__ == "__main__":
  if NUM_HANDS <= 2 and NUM_HANDS > 0:
    while(True):
      get_hand_info()
  else:
    print("Error: Number of hands not supported")


'''
Interface
L/R             -> boolean
angle_x         -> double
angle_y         -> double
angle_z         -> double
closed          -> double
L/R             -> boolean
angle_x         -> double
angle_y         -> double
angle_z         -> double
closed          -> double
'''
