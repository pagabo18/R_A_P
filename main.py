from abc import ABC, abstractmethod
import mediapipe as mp
import cv2
import numpy as np
from math import sqrt
import open3d as o3d
import mediapipe as mp


from Hands.hand_detector import hand_detector
from Capture.capture import *
from Object.Object_viewer import *

if __name__ == "__main__":

    
    objectreadfile = input("Ingrese el nombre del archivo 3D:  ")
    isfullscreen = input("¿Ejecutar en Fullscreen?  SI/NO:  ")
    isoptimized = input("¿Optimizar (solo en windows)?  SI/NO:  ")
    
    makefullscreen = False
    if isfullscreen == "SI":
        makefullscreen = True
        
    makeoptimize = False
    if isoptimized == "SI":
        makeoptimize = True
    
    hands_detection = hand_detector()
    captured = Capture(makeoptimize=makeoptimize)
    viewer = ObjectViewer(objectreadfile, makefullscreen=makefullscreen)


    with hands_detection.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while captured.cap.isOpened:
            ret, frame = captured.cap.read()
            hands_detection.get_frame(frame,hands)
            hands_detection.same_hand_detected()
            hands_detection.multi_handedness(viewer)
            viewer.make_fullscreen(makefullscreen, hands_detection)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

    captured.cap.release()
    cv2.destroyAllWindows()