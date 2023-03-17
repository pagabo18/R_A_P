
import mediapipe as mp
import cv2
import numpy as np
from math import sqrt
import open3d as o3d
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class hand_detector:
    def __init__(self):
        self.totalHands = 0
        self.moveX = 0
        self.moveY = 0
        self.moveZ = 0
        self.newZ = True
        self.refZ = 0
        self.absZ = 0
        self.initialpose = True
        
    def calc_distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def get_frame(self, frame):
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frameWidth = self.image.shape[1]
        self.frameHeight = self.image.shape[0]
        self.image = cv2.flip(self.image, 1)
        self.image.flags.writeable = False
        self.results = hands.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
    
    def Start(self):
        pass

class capture:
    def __init__(self, makeoptimize=False):
        if makeoptimize:
            self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        

class ObjectViewer:

    def __init__(self, objectreadfile, makefullscreen=False, width=1366, height=768):
        self.mesh = o3d.io.read_triangle_mesh(objectreadfile)
        self.mesh.compute_vertex_normals()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height)
        
        if makefullscreen:
            self.vis.set_full_screen(True)
        self.vis.add_geometry(self.mesh)
        self.vis.get_render_option().load_from_json("render_options.json")
        self.vis.poll_events()
        self.vis.update_renderer()
        self.zoomcounter = 0

    def vis_zoom(self, absZ):
        self.vis.get_view_control().set_zoom(absZ)
        self.vis.poll_events()
        self.vis.update_renderer()

    def vis_rotate_reset(self):
        self.vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)

    def vis_rotate(self, deltaX, deltaY):
        self.vis.get_view_control().rotate(-deltaX*10, -deltaY*10, xo=0.0, yo=0.0)
        self.vis.poll_events()
        self.vis.update_renderer()

    def vis_general_reset(self):
        self.vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)
        self.zoomcounter = self.zoomcounter + 1
        if self.zoomcounter > 1000:
            self.zoomcounter = 0
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        while True:
            self.vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)
            self.zoomcounter += 1
            if self.zoomcounter > 1000:
                self.zoomcounter = 0
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
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
    captured = capture(makeoptimize=makeoptimize)
    viewer = ObjectViewer(objectreadfile, makefullscreen=makefullscreen)
    



    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while captured.cap.isOpened():
            
            ret, frame = captured.cap.read()
            hands_detection.get_frame(frame)
            

            if  hands_detection.results.multi_handedness:
                hands_detection.totalHands = len( hands_detection.results.multi_handedness)
                if (hands_detection.totalHands == 2):
                    if ( hands_detection.results.multi_handedness[0].classification[0].label ==  hands_detection.results.multi_handedness[1].classification[0].label):
                        hands_detection.totalHands = 1
        

            if  hands_detection.results.multi_hand_landmarks:
                if hands_detection.initialpose:
                    hands_detection.initialpose = False
                    
                if (hands_detection.totalHands == 1):
                    for num, hand in enumerate( hands_detection.results.multi_hand_landmarks):
                        normalizedLandmark =  hands_detection.results.multi_hand_landmarks[
                            0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                            normalizedLandmark.x, normalizedLandmark.y,  hands_detection.frameWidth,  hands_detection.frameHeight)

                        indexTip =  hands_detection.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            indexTip.x, indexTip.y,  hands_detection.frameWidth,  hands_detection.frameHeight)

                        thumbTip =  hands_detection.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            thumbTip.x, thumbTip.y,  hands_detection.frameWidth,  hands_detection.frameHeight)

                        if pixelCoordinatesLandmark and indexTipXY and thumbTipXY is not None:
                            indexXY = (indexTipXY[0], indexTipXY[1])
                            thumbXY = (thumbTipXY[0], thumbTipXY[1])
                            cv2.circle( hands_detection.image, indexXY, 10, (255, 0, 0), 2)
                            cv2.circle( hands_detection.image, thumbXY, 10, (255, 0, 0), 2)
                            dist = hand_detector.calc_distance(indexXY, thumbXY)
                            if (dist < 50):
                                netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                                netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                                cv2.circle( hands_detection.image, (netX, netY),10, (0, 255, 0), 2)
                                deltaX = hands_detection.moveX - netX
                                hands_detection.moveX = netX
                                deltaY = hands_detection.moveY - netY
                                hands_detection.moveY = netY
                                if abs(deltaX) > 40 or abs(deltaY) > 40:
                                    print("Max reached: " +
                                    str(deltaX)+","+str(deltaY))
                                else:
                                    print(str(deltaX)+","+str(deltaY))
                                    viewer.vis_rotate(deltaX, deltaY)
                            else:
                                hands_detection.moveX = 0
                                hands_detection.moveY = 0

                        mp_drawing.draw_landmarks( hands_detection.image, hand, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2))

                elif (hands_detection.totalHands == 2):

                    handX = [0, 0]
                    handY = [0, 0]
                    isHands = [False, False]

                    for num, hand in enumerate( hands_detection.results.multi_hand_landmarks):

                        indexTip =  hands_detection.results.multi_hand_landmarks[num].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            indexTip.x, indexTip.y,  hands_detection.frameWidth,  hands_detection.frameHeight)

                        thumbTip =  hands_detection.results.multi_hand_landmarks[num].landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            thumbTip.x, thumbTip.y,  hands_detection.frameWidth,  hands_detection.frameHeight)

                        if indexTip and indexTipXY and thumbTipXY is not None:
                            indexXY = (indexTipXY[0], indexTipXY[1])
                            thumbXY = (thumbTipXY[0], thumbTipXY[1])
                            cv2.circle( hands_detection.image, indexXY, 10, (255, 0, 0), 2)
                            cv2.circle( hands_detection.image, thumbXY, 10, (255, 0, 0), 2)
                            dist = hand_detector.calc_distance(indexXY, thumbXY)
                            if (dist < 50):
                                netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                                netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                                handX[num] = netX
                                handY[num] = netY
                                isHands[num] = True

                        mp_drawing.draw_landmarks( hands_detection.image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2)
                        )

                        if (isHands[0] and isHands[1]):
                            distpar = hand_detector.calc_distance(
                                (handX[0], handY[0]), (handX[1], handY[1]))
                            if (hands_detection.newZ):
                                hands_detection.newZ = False
                                hands_detection.moveZ = distpar
                                hands_detection.refZ = distpar
                            netX = round((handX[0]+handX[1])/2)
                            netY = round((handY[0]+handY[1])/2)
                            deltaZ = (distpar - hands_detection.moveZ)/hands_detection.refZ
                            if (deltaZ < abs(1)):
                                hands_detection.absZ = hands_detection.absZ - deltaZ
                                if (hands_detection.absZ > 2.0):
                                    hands_detection.absZ = 2.0
                                elif (hands_detection.absZ < 0.5):
                                    hands_detection.absZ = 0.5
                                hands_detection.moveZ = distpar
                                print(hands_detection.absZ)
                                cv2.circle( hands_detection.image, (netX, netY),10, (0, 0, 255), 2)
                                viewer.vis_zoom(hands_detection.absZ)

                        elif (not isHands[0] and not isHands[1]):
                            hands_detection.newZ = True

            else:
                if hands_detection.initialpose == False:
                    hands_detection.initialpose = True
                    print("Regresando a posición Inicial")
                    viewer.vis_general_reset()

            if not makefullscreen:
                cv2.imshow('Hand Tracking',  hands_detection.image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    captured.cap.release()
    cv2.destroyAllWindows()
