from abc import ABC, abstractmethod
import mediapipe as mp
import cv2
import numpy as np
from math import sqrt
import open3d as o3d
import mediapipe as mp


from Hands.hand_detector import  *
from Capture.capture import *
from Object.Object_viewer import *

class RunAR:
    """
    Clase que permite iniciar la detección y visualización de objetos en 3D
    """

    def __init__(self):
        """
        Constructor de la clase RunAR
        """
        self.objectreadfile = input("Ingrese el nombre del archivo 3D: ")
        self.isfullscreen = input("¿Ejecutar en Fullscreen? SI/NO: ")
        self.isoptimized = input("¿Optimizar (solo en windows)? SI/NO: ")
        self.makefullscreen = self.isfullscreen == "SI"
        self.makeoptimize = self.isoptimized == "SI"
        self.hands_detection = hand_detector()
        self.captured = Capture(makeoptimize=self.makeoptimize)
        self.viewer = ObjectViewer(self.objectreadfile, makefullscreen=self.makefullscreen)

    def run(self):
        """
        Método que ejecuta la detección y visualización de objetos en 3D
        """
        with self.hands_detection.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while self.captured.cap.isOpened:
                ret, frame = self.captured.cap.read()
                self.hands_detection.get_frame(frame, hands)
                self.hands_detection.same_hand_detected()
                self.hands_detection.multi_handedness(self.viewer)
                self.viewer.make_fullscreen(self.makefullscreen, self.hands_detection)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        self.captured.cap.release()
        cv2.destroyAllWindows()