
import cv2
import mediapipe as mp
import open3d as o3d
from math import sqrt

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class Capture(metaclass=Singleton):
    def __init__(self, makeoptimize=False):
        if makeoptimize:
            self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def calc_distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def run_hand_tracking(self, captured, makefullscreen=False, viewer=None):

        with self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while captured.cap.isOpened():
                self.ret, self.frame = captured.cap.read()
                self.get_frame(self.frame)
                self.same_hand_detected()
                self.multi_handedness(self.viewer)

                if not makefullscreen:
                    cv2.imshow('Hand Tracking',  self.image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def get_frame(self, frame, hands):
        
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frameWidth = self.image.shape[1]
        self.frameHeight = self.image.shape[0]
        self.image = cv2.flip(self.image, 1)
        self.image.flags.writeable = False
        self.results = hands.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def index(self):

        indexTip = self.results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        indexTipXY = self.mp_drawing._normalized_to_pixel_coordinates(
            indexTip.x, indexTip.y,  self.frameWidth,  self.frameHeight)
        return indexTipXY

    def thumb(self):

        thumbTip = self.results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumbTipXY = self.mp_drawing._normalized_to_pixel_coordinates(
            thumbTip.x, thumbTip.y,  self.frameWidth,  self.frameHeight)
        return thumbTipXY

    def same_hand_detected(self):

        if self.results.multi_handedness:
            self.totalHands = len(self.results.multi_handedness)
            if (self.totalHands == 2):
                if (self.results.multi_handedness[0].classification[0].label == self.results.multi_handedness[1].classification[0].label):
                    self.totalHands = 1

    def multi_handedness(self, viewer):

        if not self.results.multi_hand_landmarks:
            return
        if self.initialpose:
            self.initialpose = False
        self.process_hand_data(viewer)

    def process_hand_data(self, viewer):

        if self.totalHands == 1:
            self.process_single_hand(viewer)
        elif self.totalHands == 2:
            self.process_double_hands(viewer)
    
    def process_single_hand(self, viewer):

        for num, hand in enumerate(self.results.multi_hand_landmarks):
            indexTipXY, thumbTipXY = self.get_finger_tips(hand)

            if indexTipXY and thumbTipXY:
                self.draw_finger_circles(indexTipXY, thumbTipXY)
                dist = hand_detector.calc_distance(indexTipXY, thumbTipXY)
                if dist < 50:
                    deltaX, deltaY = self.calculate_movement(indexTipXY, thumbTipXY)
                    netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                    netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                    cv2.circle(self.image, (netX, netY), 10, (0, 255, 0), 2)
                    self.handle_movement(viewer, deltaX, deltaY)
                    
                else:
                    self.reset_movement()

            self.draw_hand_landmarks(hand)

    def get_finger_tips(self, hand):

        indexTip = self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        indexTipNorm = hand.landmark[indexTip]
        indexTipPix = self.mp_drawing._normalized_to_pixel_coordinates(
            indexTipNorm.x, indexTipNorm.y,  self.frameWidth,  self.frameHeight)

        thumbTip = self.mp_hands.HandLandmark.THUMB_TIP
        thumbTipNorm = hand.landmark[thumbTip]
        thumbTipPix = self.mp_drawing._normalized_to_pixel_coordinates(
            thumbTipNorm.x, thumbTipNorm.y,  self.frameWidth,  self.frameHeight)

        return indexTipPix, thumbTipPix

    def draw_finger_circles(self, indexTipXY, thumbTipXY):

        cv2.circle(self.image, indexTipXY, 10, (255, 0, 0), 2)
        cv2.circle(self.image, thumbTipXY, 10, (255, 0, 0), 2)

    def calculate_movement(self, indexTipXY, thumbTipXY):

        netX = round((indexTipXY[0]+thumbTipXY[0])/2)
        netY = round((indexTipXY[1]+thumbTipXY[1])/2)
        deltaX = self.moveX - netX
        self.moveX = netX
        deltaY = self.moveY - netY
        self.moveY = netY
        return deltaX, deltaY

    def handle_movement(self, viewer, deltaX, deltaY):

        if abs(deltaX) > 40 or abs(deltaY) > 40:
            print("Max reached: " + str(deltaX)+","+str(deltaY))
        else:
            print(str(deltaX)+","+str(deltaY))
            viewer.vis_rotate(deltaX, deltaY)

    def reset_movement(self):

        self.moveX = 0
        self.moveY = 0

    def draw_hand_landmarks(self, hand):

        self.mp_drawing.draw_landmarks(
            self.image, hand, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

    def get_landmark_xy(self, hand, landmark):
 
        landmarkX = int(hand.landmark[landmark].x * self.frameWidth)
        landmarkY = int(hand.landmark[landmark].y * self.frameHeight)
        return (landmark, (landmarkX, landmarkY))

    def process_double_hands(self, viewer):

        handX = [0, 0]
        handY = [0, 0]
        isHands = [False, False]

        for num, hand in enumerate(self.results.multi_hand_landmarks):
            self.indexTip, indexTipXY = self.get_landmark_xy(hand, self.mp_hands.HandLandmark.INDEX_FINGER_TIP)
            self.thumbTip, thumbTipXY = self.get_landmark_xy(hand, self.mp_hands.HandLandmark.THUMB_TIP)

            if indexTipXY and thumbTipXY is not None:
                indexXY = (indexTipXY[0], indexTipXY[1])
                thumbXY = (thumbTipXY[0], thumbTipXY[1])
                cv2.circle(self.image, indexXY, 10, (255, 0, 0), 2)
                cv2.circle(self.image, thumbXY, 10, (255, 0, 0), 2)
                dist = hand_detector.calc_distance(indexXY, thumbXY)
                if (dist < 50):
                    netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                    netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                    handX[num] = netX
                    handY[num] = netY
                    isHands[num] = True

            self.mp_drawing.draw_landmarks(
                self.image, hand, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

        self.zoom_viewer(isHands, handX, handY, viewer)

    def zoom_viewer(self, isHands, handX, handY, viewer):
        if (isHands[0] and isHands[1]):
            distpar = hand_detector.calc_distance((handX[0], handY[0]), (handX[1], handY[1]))
            if (self.newZ):
                self.newZ = False
                self.moveZ = distpar
                self.refZ = distpar
            netX = round((handX[0]+handX[1])/2)
            netY = round((handY[0]+handY[1])/2)
            deltaZ = (distpar - self.moveZ)/self.refZ
            if (deltaZ < abs(1)):
                self.absZ = self.absZ - deltaZ
                if (self.absZ > 2.0):
                    self.absZ = 2.0
                elif (self.absZ < 0.5):
                    self.absZ = 0.5
                self.moveZ = distpar
                print(self.absZ)
                cv2.circle(self.image, (netX, netY), 10, (0, 0, 255), 2)
                viewer.vis_zoom(self.absZ)
            elif (not isHands[0] and not isHands[1]):
                self.newZ = True
            else:
                if self.initialpose == False:
                    self.initialpose = True
                    print("Regresando a posición Inicial")
                    
class ObjectViewer(metaclass=Singleton):
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
            
    def make_fullscreen(self,makefullscreen=False,hands_detection = None):
        if not makefullscreen:
                cv2.imshow('Hand Tracking', hands_detection.image)           
    def close(self):
        self.vis.destroy_window()


class RunAR:
    def __init__(self):
        self.objectreadfile = input("Ingrese el nombre del archivo 3D: ")
        self.isfullscreen = input("¿Ejecutar en Fullscreen? SI/NO: ")
        self.isoptimized = input("¿Optimizar (solo en windows)? SI/NO: ")
        self.makefullscreen = self.isfullscreen == "SI"
        self.makeoptimize = self.isoptimized == "SI"
        self.hands_detection = hand_detector()
        self.captured = Capture(makeoptimize=self.makeoptimize)
        self.viewer = ObjectViewer(self.objectreadfile, makefullscreen=self.makefullscreen)

    def run(self):
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

if __name__ == "__main__":

    inicio = RunAR()
    inicio.run()
