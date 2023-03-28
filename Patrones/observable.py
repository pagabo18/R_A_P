import cv2
import mediapipe as mp
import open3d as o3d

class Observable:
    """
    Clase Observable que maneja la lista de observadores y notifica a todos los observadores
    cada vez que se actualiza la vista
    """
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update()
