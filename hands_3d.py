import open3d as o3d
import mediapipe as mp
import cv2
from math import sqrt


    
# clase objeto
# objectreadfile = input("Ingrese el nombre del archivo 3D:  ")
objectreadfile = "dragon.ply"
objectcolor = input("Ingrese el color en R,G,B (max 255,255,255) : ")
objectcolor = objectcolor.split(",")

mesh = o3d.io.read_triangle_mesh(objectreadfile)
mesh.paint_uniform_color([int(objectcolor[0])/255,int(objectcolor[1])/255,int(objectcolor[2])/255])
mesh.compute_vertex_normals()

#clase visualizaer 
vis = o3d.visualization.Visualizer()
vis.create_window(width=1366, height=768)

isfullscreen = input("¿Ejecutar en Fullscreen?  SI/NO:  ")
makefullscreen = False
if isfullscreen == "SI":
    makefullscreen = True


# isoptimized = input("¿Optimizar (solo en windows)?  SI/NO:  ")
# makeoptimize = False
# if isoptimized == "SI":
#     makeoptimize = True
    
# if makefullscreen:
#     vis.set_full_screen(True)  

vis.add_geometry(mesh)
vis.get_render_option().load_from_json("render_options.json")
vis.poll_events()
vis.update_renderer()

print("Ejecutando...")

# if makeoptimize:
#     cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
# else:
#     cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


#clase hand detector 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#These variables store the amount of motion detected in the user’s hand on the X, Y and Z axis.
moveX = 0
moveY = 0
moveZ = 0

#It is a Boolean variable that indicates whether a significant change in the user’s hand position on the Z axis has been detected.
newZ = True

#It is a variable that stores the reference position of the user’s hand on the Z axis.
refZ = 0
#It is a variable that stores the absolute position of the user’s hand on the Z axis.
absZ = 0

#It is a boolean variable that indicates whether the current position of the user’s hand should be considered as an initial position for tracking.
initialpose = True

#It is a variable used to keep track of how many times a zoom gesture has been detected by the user.
zoomcounter = 0

#calcula la distancia entre dos puntos en un plano
def calc_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

#Enter a code block that initializes 
#hand detection using the mediapipe library. 
#A Hands object with minimal trust parameters is used for both hand detection and hand tracking.
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        
        #EDIT IMAGE
        #Capture a frame of the video. 
        #The ret variable will be true if a frame could be read correctly and frame will be the captured image.
        ret, frame = cap.read()
        #Converts the captured image to RGB so that it can be processed by mediapipe.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #gets the dimensions of the image
        frameWidth = image.shape[1]
        frameHeight = image.shape[0]
        #Flips the captured image horizontally. 
        #This is necessary for the right hand movement on the screen to match the right hand movement of the user.
        image = cv2.flip(image, 1)
        #The image is marked as not writable to pass it to mediapipe for processing faster.
        image.flags.writeable = False
        #Processes the image to detect the position of hands using the hands object
        results = hands.process(image)
        
        #CLOSE EDIT IMAGE
        #The image is transformed to its original variables
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pos = (0,0)
        cv2.rectangle(image, pos, (frameWidth, frameHeight),(0, 0, 0), -1)

        #number of hands detected
        totalHands = 0

        #Same Hands detected
        if results.multi_handedness:
            totalHands = len(results.multi_handedness)
            if(totalHands == 2):
                if(results.multi_handedness[0].classification[0].label == results.multi_handedness[1].classification[0].label):
                    totalHands = 1
        
        #Draws the detected hands on the screen
        if results.multi_hand_landmarks:
            
            #Reset initial pose Vis
            if initialpose:
                initialpose = False
            
            #One Hand Detected
            if(totalHands == 1):
                for num, hand in enumerate(results.multi_hand_landmarks):
                    #create a method landmark
                    #create method normalize to pixel
                    
                    #object and contains the normalized (i.e., in the range of [0, 1]) x, y, and z coordinates of the landmark relative to the detected hand.
                    normalizedLandmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    #is a tuple containing the x and y pixel coordinates of the landmark in the current frame.
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
                    
                    
                    #index landmark coordinates converted to pixel
                    indexTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    indexTipXY = mp_drawing._normalized_to_pixel_coordinates(indexTip.x, indexTip.y, frameWidth, frameHeight)

                    #thumb landmark coordinates converted to pixel
                    thumbTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(thumbTip.x, thumbTip.y, frameWidth, frameHeight)

                    #This line checks whether all three landmarks have been detected in the current frame.
                    if pixelCoordinatesLandmark and indexTipXY and thumbTipXY is not None:
                        #gets the coordinates of the index and thumb to circle them 
                        indexXY = (indexTipXY[0],indexTipXY[1])
                        thumbXY = (thumbTipXY[0],thumbTipXY[1])
                        cv2.circle(image, indexXY, 10, (255, 0, 0), 2)
                        cv2.circle(image, thumbXY, 10, (255, 0, 0), 2)
                        
                        #distance between fingers
                        dist = calc_distance(indexXY,thumbXY)
                        
                        #Method distance
                        if(dist < 50):
                            #get the center coordinates of the finger and the thumb
                            netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                            netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                            
                            #draw a circle in those coordinates 
                            cv2.circle(image, (netX,netY), 10, (0, 255, 0), 2)
                            
                            #calculate the differences between coordinates from the last position and the new one
                            deltaX = moveX - netX
                            moveX = netX
                            deltaY = moveY - netY
                            moveY = netY
                            
                            #calculate the max distance permited to create the circle
                            if abs(deltaX) > 40 or abs(deltaY) > 40:
                                print("Max reached: "+str(deltaX)+","+str(deltaY))
                            else: 
                                print(str(deltaX)+","+str(deltaY))
                                
                                #Object rotate
                                vis.get_view_control().rotate(-deltaX*10, -deltaY*10, xo=0.0, yo=0.0)
                                vis.poll_events()
                                vis.update_renderer()
                        else:
                            moveX = 0
                            moveY = 0
                    #create a group of landmarks and connectios for each dtected hand
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
        elif(totalHands == 2):
                #xy positions of right and left hand
                handX = [0,0]
                handY = [0,0]
                
                #if hand is detected
                isHands = [False, False] 

                #for each hand
                for num, hand in enumerate(results.multi_hand_landmarks):
                    
                    #create landmarks od index and thumb
                    indexTip = results.multi_hand_landmarks[num].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    indexTipXY = mp_drawing._normalized_to_pixel_coordinates(indexTip.x, indexTip.y, frameWidth, frameHeight)

                    thumbTip = results.multi_hand_landmarks[num].landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumbTipXY = mp_drawing._normalized_to_pixel_coordinates(thumbTip.x, thumbTip.y, frameWidth, frameHeight)

                    #This line checks whether all three landmarks have been detected in the current frame.
                    if indexTip and indexTipXY and thumbTipXY is not None:
                        #get the coordinates of each finger
                        indexXY = (indexTipXY[0],indexTipXY[1])
                        thumbXY = (thumbTipXY[0],thumbTipXY[1])
                        
                        #draw a circle in those coordinates
                        cv2.circle(image, indexXY, 10, (255, 0, 0), 2)
                        cv2.circle(image, thumbXY, 10, (255, 0, 0), 2)
                        
                        #calculate distance between the two fingers
                        dist = calc_distance(indexXY,thumbXY)
                        #if the distance is les 50 
                        if(dist < 50):
                            #draw a circle in the center of the two fingers
                            netX = round((indexTipXY[0]+thumbTipXY[0])/2)
                            netY = round((indexTipXY[1]+thumbTipXY[1])/2)
                            handX[num] = netX
                            handY[num] = netY
                            isHands[num] = True
                            
                            #draw a circle in those coordinates
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

                    #two hand detected
                    if(isHands[0] and isHands[1]):
                        #calculate distance between hands
                        dist_twohands = calc_distance((handX[0],handY[0]),(handX[1],handY[1]))
                        
                        #move z could be replaced
                        if(newZ):
                            newZ = False
                            moveZ = dist_twohands
                            refZ = dist_twohands
                        
                        #calculate the center of the two hands
                        netX = round((handX[0]+handX[1])/2)
                        netY = round((handY[0]+handY[1])/2)
                        
                        #Calculate the change distance between two hands
                        deltaZ = (dist_twohands - moveZ)/refZ
                        
                        #if distance between hands is less 1
                        if(deltaZ < abs(1)):
                            #update the zoom factor 
                            absZ = absZ - deltaZ
                            
                            #limit to 2 / 0.5
                            if (absZ > 2.0):
                                absZ = 2.0
                            elif(absZ <0.5):
                                absZ = 0.5
                            
                            #update the current distance 
                            moveZ = dist_twohands
                            print(absZ)
                            
                            #draw a red circle between the two hands
                            cv2.circle(image, (netX,netY), 10, (0, 0, 255), 2)
                            
                            #update the zoom with the absolute distance
                            vis.get_view_control().set_zoom(absZ)
                            vis.poll_events()
                            vis.update_renderer()
                    #see if there is one hand
                    elif (not isHands[0] and not isHands[1]):
                        newZ = True
        else: 
            #resetea la rotación y zoom inicial
            if not initialpose:
                initialpose = True
                print("Regresando a posición Inicial")
                vis.get_view_control().set_zoom(1)
            
            vis.get_view_control().rotate(5, 0, xo=0.0, yo=0.0)
            zoomcounter = zoomcounter + 1 
            if zoomcounter > 1000:
                zoomcounter = 0
            vis.poll_events()
            vis.update_renderer()

        
        # if not makefullscreen:
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#close
cap.release()
cv2.destroyAllWindows()



           