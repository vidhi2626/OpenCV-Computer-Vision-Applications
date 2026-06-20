import cv2 
 
# Load Haar cascades (Ensure the correct path) 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml') 
 
# Read the image 
img = cv2.imread('woman.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
# Detect faces 
faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
 
for (x, y, w, h) in faces: 
    roi_gray = gray[y:y + h, x:x + w] 
    roi_color = img[y:y + h, x:x + w] 
 
    noses = nose_cascade.detectMultiScale(roi_gray, 1.1, 4) 
    for (nx, ny, nw, nh) in noses: 
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2) 
 
# Display the result 
cv2.imshow('Nose Detection', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
