import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
#cv2_imshow(img)

cap = cv2.imread('photo.jpg', cv2.IMREAD_UNCHANGED)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: #You can pass `max_num_hands` argument here as well if you want to detect more that one hand
        
        image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        
        results = hands.process(image)
        
        image.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                         )
            
        cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        #cv2_imshow(image)

cv2.destroyAllWindows()


