import mediapipe as mp
import cv2
from math import sqrt
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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
    def calc_distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    
    def run_hand_tracking(self, captured, makefullscreen=False, viewer=None):
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while captured.cap.isOpened():
                self.ret, self.frame = captured.cap.read()
                self.get_frame(self.frame)
                self.same_hand_detected()
                self.multi_handedness(self.viewer)

                if not makefullscreen:
                    cv2.imshow('Hand Tracking',  self.image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def get_frame(self, frame,hands):
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
    
    def index(self):
        indexTip =  self.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(indexTip.x, indexTip.y,  self.frameWidth,  self.frameHeight)
        return indexTipXY
    
    def thumb(self):
        thumbTip =  self.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates( thumbTip.x, thumbTip.y,  self.frameWidth,  self.frameHeight)
        return thumbTipXY
    
    def same_hand_detected(self):
        if  self.results.multi_handedness:
                self.totalHands = len( self.results.multi_handedness)
                if (self.totalHands == 2):
                    if ( self.results.multi_handedness[0].classification[0].label ==  self.results.multi_handedness[1].classification[0].label):
                        self.totalHands = 1
    
    def multi_handedness(self,viewer):
        if  self.results.multi_hand_landmarks:
                if self.initialpose:
                    self.initialpose = False   
                if (self.totalHands == 1):
                    for num, hand in enumerate( self.results.multi_hand_landmarks):
                        normalizedLandmark =  self.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y,  self.frameWidth,  self.frameHeight)

                        indexTipXY = self.index()

                        thumbTipXY = self.thumb()

                        if pixelCoordinatesLandmark and indexTipXY and thumbTipXY is not None:
                            indexXY = (indexTipXY[0], indexTipXY[1])
                            thumbXY = (thumbTipXY[0], thumbTipXY[1])
                            cv2.circle( self.image, indexXY, 10, (255, 0, 0), 2)
                            cv2.circle( self.image, thumbXY, 10, (255, 0, 0), 2)
                            dist = hand_detector.calc_distance(indexXY, thumbXY)
                            if (dist < 50):
                                netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                                netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                                cv2.circle( self.image, (netX, netY),10, (0, 255, 0), 2)
                                deltaX = self.moveX - netX
                                self.moveX = netX
                                deltaY = self.moveY - netY
                                self.moveY = netY
                                if abs(deltaX) > 40 or abs(deltaY) > 40:
                                    print("Max reached: " +
                                    str(deltaX)+","+str(deltaY))
                                else:
                                    print(str(deltaX)+","+str(deltaY))
                                    viewer.vis_rotate(deltaX, deltaY)
                            else:
                                self.moveX = 0
                                self.moveY = 0

                        mp_drawing.draw_landmarks( self.image, hand, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2))

                elif (self.totalHands == 2):

                    handX = [0, 0]
                    handY = [0, 0]
                    isHands = [False, False]

                    for num, hand in enumerate( self.results.multi_hand_landmarks):

                        indexTip =  self.results.multi_hand_landmarks[num].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            indexTip.x, indexTip.y,  self.frameWidth,  self.frameHeight)

                        thumbTip =  self.results.multi_hand_landmarks[num].landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(
                            thumbTip.x, thumbTip.y,  self.frameWidth,  self.frameHeight)

                        if indexTipXY and thumbTipXY is not None:
                            indexXY = (indexTipXY[0], indexTipXY[1])
                            thumbXY = (thumbTipXY[0], thumbTipXY[1])
                            cv2.circle( self.image, indexXY, 10, (255, 0, 0), 2)
                            cv2.circle( self.image, thumbXY, 10, (255, 0, 0), 2)
                            dist = hand_detector.calc_distance(indexXY, thumbXY)
                            if (dist < 50):
                                netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                                netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                                handX[num] = netX
                                handY[num] = netY
                                isHands[num] = True

                        mp_drawing.draw_landmarks( self.image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(
                        color=(250, 44, 250), thickness=2, circle_radius=2)
                        )

                        if (isHands[0] and isHands[1]):
                            distpar = hand_detector.calc_distance(
                                (handX[0], handY[0]), (handX[1], handY[1]))
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
                                cv2.circle( self.image, (netX, netY),10, (0, 0, 255), 2)
                                viewer.vis_zoom(self.absZ)

                        elif (not isHands[0] and not isHands[1]):
                            self.newZ = True 
                            
                        else:
                            if self.initialpose == False:
                                self.initialpose = True
                                print("Regresando a posición Inicial")
                                viewer.vis_general_reset()    
    