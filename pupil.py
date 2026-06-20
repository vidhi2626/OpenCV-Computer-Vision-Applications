import cv2 
import numpy as np 
# Load Haar cascade for face and eyes 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
 
# Read the image 
image = cv2.imread('girl2.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Detect faces 
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) 
 
for (x, y, w, h) in faces: 
    face_roi = gray[y:y+h, x:x+w] 
    eyes = eye_cascade.detectMultiScale(face_roi) 
 
    for (ex, ey, ew, eh) in eyes: 
        eye_roi = face_roi[ey:ey+eh, ex:ex+ew] 
        blurred_eye = cv2.GaussianBlur(eye_roi, (9, 9), 0)  # Reduce noise 
         
        # Use HoughCircles to detect pupils 
        circles = cv2.HoughCircles(blurred_eye, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, 
                                   param1=60, param2=20, minRadius=5, maxRadius=30) 
         
        if circles is not None: 
            circles = np.uint16(np.around(circles)) 
            for circle in circles[0, :]: 
                cx, cy, radius = circle 
                cv2.circle(image[y+ey:y+ey+eh, x+ex:x+ex+ew], (cx, cy), radius, (0, 255, 0), 2) 
 
# Display the output 
cv2.imshow('Pupil Detection', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
