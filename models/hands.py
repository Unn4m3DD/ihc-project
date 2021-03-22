from math import atan2, degrees, radians, sqrt
import mediapipe as mp
mp = mp.solutions.hands

class Hand:
    def __init__(self, classification = None):
        self.mp_hands = mp
        self.classification = classification # has label and score attributes
        self.angle_x = None
        self.angle_y = None
        self.angle_z = None
        self.closed = None
        self.hand_landmarks = None
        
    def hand_calculations(self, hand_landmarks):
        if hand_landmarks:
            self.hand_landmarks = hand_landmarks
            
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_insertion = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_insertion = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            pinky_finger_insertion = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_finger_insertion = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

            self.angle_z = -degrees(atan2(
                (wrist.z - middle_finger_tip.z),
                (
                    sqrt(
                        (wrist.x - middle_finger_tip.x) ** 2 +
                        (wrist.y - middle_finger_tip.y) ** 2
                    )
                )
            ))

            self.angle_x = degrees(atan2(
                (pinky_finger_insertion.z - index_finger_insertion.z),
                (
                    sqrt(
                        (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 +
                        (pinky_finger_insertion.y - index_finger_insertion.y) ** 2
                    )
                )
            ))

            self.angle_y = degrees(atan2(
                (middle_finger_insertion.y - wrist.y),
                (middle_finger_insertion.x - wrist.x)
            ))

            self.closed = (sqrt(
                (ring_finger_tip.x - wrist.x) ** 2 +
                (ring_finger_tip.y - wrist.y) ** 2 +
                (ring_finger_tip.z - wrist.z) ** 2
            ) / sqrt(
                (pinky_finger_insertion.x - index_finger_insertion.x) ** 2 +
                (pinky_finger_insertion.y - index_finger_insertion.y) ** 2 +
                (pinky_finger_insertion.z - index_finger_insertion.z) ** 2 * 0.5
            ) ) / 2.5 - .45
            
            if(self.closed > 1):
                self.closed = 1
            if(self.closed < 0):
                self.closed = 0 