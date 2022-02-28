from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
from pynput.keyboard import Key,Controller
import time
from PIL import Image

keyboard = Controller()

cap = cv2.VideoCapture(0)

# Anpassen der Größe des Kamera-Anzeigefensters
# cap.set(3, 1280)
# cap.set(4, 720)

# Detector für Hände
detector = HandDetector(detectionCon=0.8, maxHands=2) # detection confidence, maximale Anzahl an Händen

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # draw=False, falls die erkannte Hand im Kamerafenster nicht eingezeichnet werden soll.

    # Schriftfarbe usw.
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (255, 0, 0)
    thickness = 2

    # Wenn hand erkannt wird
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # Liste mit 21 Punkten innerhalb der erkannten Hand
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # Zentrum der Hand
        handType1 = hand1["type"]  # Links oder Rechts (left or right)

        # Zähler, wie viele Finger die Hand anzeigt
        fingers1 = detector.fingersUp(hand1)
        fingersup1 = np.count_nonzero(fingers1)
        fingersup1_output = 'Fingers up ' + handType1 + ': ' + str(fingersup1)
        org = (20, 50) # Ort, an dem Text angezeigt werden soll
        cv2.putText(img, fingersup1_output, org, font, fontScale, color, thickness, cv2.LINE_AA)

        length_ti, info, img = detector.findDistance(lmList1[8], lmList1[4], img)  # Daumen und Zeigefinger (Abstand)
        length_reference, info_reference, img = detector.findDistance(lmList1[0], lmList1[17], img)  # Zentrum zu Ende Mittelhandknochen kleiner Finger
        
        # Länge Abstand Daumen zu Zeigefinger in Pixeln anzeigen
        length_output = "Distance in pixels: " + str(length_ti)
        org = (20, 450)
        cv2.putText(img, length_output, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Weitere Abstände ermitteln
        length_ci, info_ci, img = detector.findDistance(lmList1[8], lmList1[0], img) # Zentrum zu Zeigefinger
        length_cm, info_cm, img = detector.findDistance(lmList1[12], lmList1[0], img)  # Daumen und Mittelfinger
        length_cr, info_cr, img = detector.findDistance(lmList1[16], lmList1[0], img)  # Daumen und Ringfinger
        length_cp, info_cp, img = detector.findDistance(lmList1[20], lmList1[0], img)  # Daumen und Kleiner Finger
        length_tm, info_tm, img = detector.findDistance(lmList1[12], lmList1[4], img)  # Daumen und Mittelfinger
        length_tr, info2_tr, img = detector.findDistance(lmList1[16], lmList1[4], img)  # Daumen und Mittelfinger
        
        # Lauter
        if length_ti*6.2 < length_reference:
            keyboard.tap(Key.media_volume_down)
            
        # Leiser
        if length_ti > length_reference*1.5:
            keyboard.tap(Key.media_volume_up)
        
        # Mute
        if length_ci > length_reference*1.7 and length_tm*5 < length_reference and length_tr*5 < length_reference and length_cp > length_reference*1.7:
            keyboard.press(Key.media_volume_mute)
            keyboard.release(Key.media_volume_mute)
            time.sleep(1)

        # Zweite Hand
        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)
            fingersup2 = np.count_nonzero(fingers2)
            fingersup2_output = 'Fingers up ' + handType2 + ': ' + str(fingersup2)

            org = (20, 80)
            cv2.putText(img, fingersup2_output, org, font, fontScale, color, thickness, cv2.LINE_AA)
         
    # Anzeigen der Kamera - kann auskommentiert werden
    cv2.imshow("Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
