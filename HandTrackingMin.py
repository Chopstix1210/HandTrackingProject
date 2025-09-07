import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(1)

# Hand detection module 
mpHands = mp.solutions.hands # type: ignore not sure why pyright is saying it's not a library 
hands = mpHands.Hands()

# draw out points on hands 
mpDraw = mp.solutions.drawing_utils # type: ignore 

# tracking fps 
pTime = 0
cTime = 0


while True: 
    success, image = cap.read()

    # send rgb image 
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # extract information to see if there are multiple hands 
    # print(results.multi_hand_landmarks) 
    if (results.multi_hand_landmarks): 
        for handLMS in results.multi_hand_landmarks: 
            for id, lm in enumerate(handLMS.landmark): 
                #print(id, lm) 
                # need to multiply width and height with the pixel value as this just gives the pixel ratio 
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                if id==0:
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(image, handLMS, mpHands.HAND_CONNECTIONS)

    # fps count 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", image)
    cv2.waitKey(1)