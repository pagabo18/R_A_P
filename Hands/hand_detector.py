import mediapipe as mp
import cv2
from math import sqrt
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class hand_detector:
    '''
    Clase que genera las detecciones de manos
    
    '''
    def __init__(self):
        '''
        Constructor de la clase hand_detector. Inicializa las variables de seguimiento de mano.
        
        '''
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
        '''
        Método estático que calcula la distancia euclidiana entre dos puntos p1 y p2.
        
        '''
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def run_hand_tracking(self, captured, makefullscreen=False, viewer=None):
        '''
        Método que ejecuta el seguimiento de mano en un flujo de video.
        
        Parametros 
        -----------
        
        captured:
            Objeto de la clase Capture que contiene el flujo de video.
        
        makefullscreen: 
            Booleano que indica si se ejecuta en modo fullscreen.
        
        viewer:
            Objeto de la clase ObjectViewer que contiene el objeto 3D.
        
        '''
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

    def get_frame(self, frame, hands):
        '''
        Método que procesa un solo cuadro de video para detectar y analizar la información de la mano
        
        Parametros:
        -----------
        
        frame:
            Cuadro de video que se procesa.
        
        hands:
            Objeto de la clase Hands que contiene la información de la mano.
        
        '''
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frameWidth = self.image.shape[1]
        self.frameHeight = self.image.shape[0]
        self.image = cv2.flip(self.image, 1)
        self.image.flags.writeable = False
        self.results = hands.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def index(self):
        '''
         Método que devuelve las coordenadas (x, y) del extremo del dedo índice de la mano detectada actualmente.
        
        '''
        indexTip = self.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        indexTipXY = mp_drawing._normalized_to_pixel_coordinates(
            indexTip.x, indexTip.y,  self.frameWidth,  self.frameHeight)
        return indexTipXY

    def thumb(self):
        '''
        Método que devuelve las coordenadas (x, y) del extremo del dedo pulgar de la mano detectada actualmente.
        
        '''
        thumbTip = self.results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(
            thumbTip.x, thumbTip.y,  self.frameWidth,  self.frameHeight)
        return thumbTipXY

    def same_hand_detected(self):
        '''
        Método que comprueba si se ha detectado la misma mano en el cuadro de video actual.
        
        '''
        if self.results.multi_handedness:
            self.totalHands = len(self.results.multi_handedness)
            if (self.totalHands == 2):
                if (self.results.multi_handedness[0].classification[0].label == self.results.multi_handedness[1].classification[0].label):
                    self.totalHands = 1

    def multi_handedness(self, viewer):
        '''
        Método que procesa la información de la mano detectada en el cuadro de video actual.
        Parametros:
        -----------
        
        viewer:
            Objeto de la clase ObjectViewer que contiene el objeto 3D.
        '''
        if not self.results.multi_hand_landmarks:
            return
        if self.initialpose:
            self.initialpose = False
        self.process_hand_data(viewer)

    def process_hand_data(self, viewer):
        '''
        Método que procesa los datos de las manos detectadas 
        para realizar el seguimiento de movimiento de la mano. 
        Si se detecta una mano, llama al método process_single_hand. 
        Si se detectan dos manos, llama al método process_double_hands.
        
        Parametros:
        -----------
        viewer:
            Objeto de la clase ObjectViewer que contiene el objeto 3D.
        
        '''
        if self.totalHands == 1:
            self.process_single_hand(viewer)
        elif self.totalHands == 2:
            self.process_double_hands(viewer)
    
    def process_single_hand(self, viewer):
        '''
        Método que realiza el seguimiento de movimiento de una mano.
        '''
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
        '''
        Método que devuelve las coordenadas (x, y) de la punta del índice y la punta del pulgar de una mano.
        parametros:
        -----------
        hand:
            Objeto de la clase HandLandmark que contiene la información de la mano.
        
        '''
        indexTip = mp_hands.HandLandmark.INDEX_FINGER_TIP
        indexTipNorm = hand.landmark[indexTip]
        indexTipPix = mp_drawing._normalized_to_pixel_coordinates(
            indexTipNorm.x, indexTipNorm.y,  self.frameWidth,  self.frameHeight)

        thumbTip = mp_hands.HandLandmark.THUMB_TIP
        thumbTipNorm = hand.landmark[thumbTip]
        thumbTipPix = mp_drawing._normalized_to_pixel_coordinates(
            thumbTipNorm.x, thumbTipNorm.y,  self.frameWidth,  self.frameHeight)

        return indexTipPix, thumbTipPix

    def draw_finger_circles(self, indexTipXY, thumbTipXY):
        '''
        Método que dibuja un círculo en la punta del índice y la punta del pulgar de una mano.
        
        parametros
        ----------
        indexTipXY:
            Tupla con las coordenadas (x, y) de la punta del índice.
        thumbTipXY:
            Tupla con las coordenadas (x, y) de la punta del pulgar.
        
        '''
        cv2.circle(self.image, indexTipXY, 10, (255, 0, 0), 2)
        cv2.circle(self.image, thumbTipXY, 10, (255, 0, 0), 2)

    def calculate_movement(self, indexTipXY, thumbTipXY):
        '''
        Método que calcula la cantidad de movimiento en las coordenadas (x, y) de la mano
        en un solo cuadro de video en función de la posición actual de la punta del índice
        y la punta del pulgar y su posición anterior. 
        
        Si se encuentra una cantidad suficiente de movimiento, 
        actualiza los valores de desplazamiento moveX y moveY y los devuelve.
        Si no se encuentra suficiente movimiento, devuelve 0 para ambos valores.
        
        parametros
        ----------
        indexTipXY:
            Tupla con las coordenadas (x, y) de la punta del índice.
        thumbTipXY:
            Tupla con las coordenadas (x, y) de la punta del pulgar.
            
        '''
        netX = round((indexTipXY[0]+thumbTipXY[0])/2)
        netY = round((indexTipXY[1]+thumbTipXY[1])/2)
        deltaX = self.moveX - netX
        self.moveX = netX
        deltaY = self.moveY - netY
        self.moveY = netY
        return deltaX, deltaY

    def handle_movement(self, viewer, deltaX, deltaY):
        '''
        Método que gestiona el movimiento de la mano en el objeto 3D.
        Si la cantidad de movimiento es suficiente, llama al método vis_rotate
        del objeto viewer para realizar el movimiento.
            
        parametros
        ----------
        viewer:
            Objeto de la clase ObjectViewer que contiene el objeto 3D.
        deltaX:
            Cantidad de movimiento en la coordenada x.
        deltaY:
            Cantidad de movimiento en la coordenada y.
        '''
        if abs(deltaX) > 40 or abs(deltaY) > 40:
            print("Max reached: " + str(deltaX)+","+str(deltaY))
        else:
            print(str(deltaX)+","+str(deltaY))
            viewer.vis_rotate(deltaX, deltaY)

    def reset_movement(self):
        '''
        Método que reinicia los valores de desplazamiento moveX y moveY.
        '''
        self.moveX = 0
        self.moveY = 0

    def draw_hand_landmarks(self, hand):
        '''
        Método que dibuja los puntos de referencia de la mano.
        parametros
        ----------
        hand:
            Objeto de la clase HandLandmark que contiene la información de la mano.
        '''
        mp_drawing.draw_landmarks(
            self.image, hand, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

    def get_landmark_xy(self, hand, landmark):
        '''
        Método que devuelve las coordenadas (x, y) de un punto de referencia de la mano.
        parametros
        ----------
        hand:
            Objeto de la clase HandLandmark que contiene la información de la mano.
        landmark:
            Punto de referencia de la mano.
        '''
        landmarkX = int(hand.landmark[landmark].x * self.frameWidth)
        landmarkY = int(hand.landmark[landmark].y * self.frameHeight)
        return (landmark, (landmarkX, landmarkY))

    def process_double_hands(self, viewer):
        '''
        Método que procesa la información de dos manos. 
        si detecta las dos manos pero no la distancia entre los pulgares no hace nada
        si detecta las dos manos y la distancia entre los pulgares es suficiente,
        crea un circulo en el centro y realiza un zoom en el objeto
        
        parametros
        ----------
        viewer:
            Objeto de la clase ObjectViewer que contiene el objeto 3D.
        '''
        handX = [0, 0]
        handY = [0, 0]
        isHands = [False, False]

        for num, hand in enumerate(self.results.multi_hand_landmarks):
            self.indexTip, indexTipXY = self.get_landmark_xy(hand, mp_hands.HandLandmark.INDEX_FINGER_TIP)
            self.thumbTip, thumbTipXY = self.get_landmark_xy(hand, mp_hands.HandLandmark.THUMB_TIP)

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

            mp_drawing.draw_landmarks(
                self.image, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

        self.zoom_viewer(isHands, handX, handY, viewer)

    def zoom_viewer(self, isHands, handX, handY, viewer):
        '''
        Método que realiza el zoom en el objeto 3D.
        Si detecta las dos manos y la distancia entre los pulgares es suficiente,
        realiza un zoom en el objeto.
        si no, regresa a la posicion inicial.
        
        parametros
        ----------
        isHands:
            Lista con los valores booleanos que indican si se ha detectado una mano.
        handX:
            Lista con las coordenadas x de las manos.
        handY:
            Lista con las coordenadas y de las manos.
        viewer:
            Objeto de la clase ObjectViewer que contiene el objeto 3D.
        '''
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