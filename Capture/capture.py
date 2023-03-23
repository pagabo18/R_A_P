import cv2

from Patrones.singleton import *

class Capture(metaclass=Singleton):
    def __init__(self, makeoptimize=False):
        '''
        Constructor de la clase Capture que utiliza la optimización solo para windows
        
        Parámetros:
        makeoptimize (bool): Indica si se desea una optimización de captura. Por defecto es False.
        
        Returns:
        None
        
        '''
        if makeoptimize:
            self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
