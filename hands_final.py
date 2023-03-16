
import mediapipe as mp
import cv2
import numpy as np
from math import sqrt
import open3d as o3d
import mediapipe as mp

# makefullscreen = False

# isoptimized = input("¿Optimizar (solo en windows)?  SI/NO:  ")
# makeoptimize = False
# if isoptimized == "SI":
#     makeoptimize = True

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

moveX = 0
moveY = 0
moveZ = 0
newZ = True
refZ = 0
absZ = 0
initialpose = True
zoomcounter = 0


class hand_detector:
    def __init__(self):
        pass
    # def calc_distance(p1, p2):
    #     return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# class capture:
#     def __init__(self,makeoptimize):
#         if makeoptimize == "SI":
#             self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
#         else:
#             self.cap = cv2.VideoCapture(0)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


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


def calc_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


if __name__ == "__main__":
    objectreadfile = input("Ingrese el nombre del archivo 3D:  ")
    isfullscreen = input("¿Ejecutar en Fullscreen?  SI/NO:  ")
    makefullscreen = False
    if isfullscreen == "SI":
        makefullscreen = True
    viewer = ObjectViewer(objectreadfile, makefullscreen=makefullscreen)

    isoptimized = input("¿Optimizar (solo en windows)?  SI/NO:  ")
    makeoptimize = False
    if isoptimized == "SI":
        makeoptimize = True

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    if makeoptimize:
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    moveX = 0
    moveY = 0
    moveZ = 0
    newZ = True
    refZ = 0
    absZ = 0
    initialpose = True
    zoomcounter = 0

    def calc_distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frameWidth = image.shape[1]
            frameHeight = image.shape[0]

            image = cv2.flip(image, 1)

            image.flags.writeable = False

            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # pos = (0,0)
            # cv2.rectangle(image, pos, (frameWidth, frameHeight),(0, 0, 0), -1)

            totalHands = 0

            if results.multi_handedness:
                totalHands = len(results.multi_handedness)
                if (totalHands == 2):
                    if (results.multi_handedness[0].classification[0].label == results.multi_handedness[1].classification[0].label):
                        totalHands = 1

            if results.multi_hand_landmarks:
                if initialpose:
                    initialpose = False
                if (totalHands == 1):
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        normalizedLandmark = results.multi_hand_landmarks[
                            0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                            normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)

                        indexTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            indexTip.x, indexTip.y, frameWidth, frameHeight)

                        thumbTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            thumbTip.x, thumbTip.y, frameWidth, frameHeight)

                        if pixelCoordinatesLandmark and indexTipXY and thumbTipXY is not None:
                            indexXY = (indexTipXY[0], indexTipXY[1])
                            thumbXY = (thumbTipXY[0], thumbTipXY[1])
                            cv2.circle(image, indexXY, 10, (255, 0, 0), 2)
                            cv2.circle(image, thumbXY, 10, (255, 0, 0), 2)
                            dist = calc_distance(indexXY, thumbXY)
                            if (dist < 50):
                                netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                                netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                                cv2.circle(image, (netX, netY),
                                           10, (0, 255, 0), 2)
                                deltaX = moveX - netX
                                moveX = netX
                                deltaY = moveY - netY
                                moveY = netY
                                if abs(deltaX) > 40 or abs(deltaY) > 40:
                                    print("Max reached: " +
                                          str(deltaX)+","+str(deltaY))
                                else:
                                    print(str(deltaX)+","+str(deltaY))
                                    viewer.vis_rotate(deltaX, deltaY)
                            else:
                                moveX = 0
                                moveY = 0

                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(
                                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                                  mp_drawing.DrawingSpec(
                                                      color=(250, 44, 250), thickness=2, circle_radius=2)
                                                  )

                elif (totalHands == 2):

                    handX = [0, 0]
                    handY = [0, 0]
                    isHands = [False, False]

                    for num, hand in enumerate(results.multi_hand_landmarks):

                        indexTip = results.multi_hand_landmarks[num].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            indexTip.x, indexTip.y, frameWidth, frameHeight)

                        thumbTip = results.multi_hand_landmarks[num].landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            thumbTip.x, thumbTip.y, frameWidth, frameHeight)

                        if indexTip and indexTipXY and thumbTipXY is not None:
                            indexXY = (indexTipXY[0], indexTipXY[1])
                            thumbXY = (thumbTipXY[0], thumbTipXY[1])
                            cv2.circle(image, indexXY, 10, (255, 0, 0), 2)
                            cv2.circle(image, thumbXY, 10, (255, 0, 0), 2)
                            dist = calc_distance(indexXY, thumbXY)
                            if (dist < 50):
                                netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                                netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                                handX[num] = netX
                                handY[num] = netY
                                isHands[num] = True

                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2)
                        )

                        if (isHands[0] and isHands[1]):
                            distpar = calc_distance(
                                (handX[0], handY[0]), (handX[1], handY[1]))
                            if (newZ):
                                newZ = False
                                moveZ = distpar
                                refZ = distpar
                            netX = round((handX[0]+handX[1])/2)
                            netY = round((handY[0]+handY[1])/2)
                            deltaZ = (distpar - moveZ)/refZ
                            if (deltaZ < abs(1)):
                                absZ = absZ - deltaZ
                                if (absZ > 2.0):
                                    absZ = 2.0
                                elif (absZ < 0.5):
                                    absZ = 0.5
                                moveZ = distpar
                                print(absZ)
                                cv2.circle(image, (netX, netY),10, (0, 0, 255), 2)
                                viewer.vis_zoom(absZ)

                        elif (not isHands[0] and not isHands[1]):
                            newZ = True

            else:
                if not initialpose:
                    initialpose = True
                    print("Regresando a posición Inicial")
                    viewer.vis_general_reset()

            if not makefullscreen:
                cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
