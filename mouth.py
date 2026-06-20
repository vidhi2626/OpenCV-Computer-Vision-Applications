
# import required libraries 
import cv2 
 
# read input image 
img = cv2.imread('mouth.jpeg') 
 
# convert the image to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
# read haar cascade for face detection 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
 
# read haar cascade for mouth detection 
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml') 
 
# Detects faces in the input image 
faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
print('Number of detected faces:', len(faces)) 
 
# loop over all the faces detected 
for (x,y,w,h) in faces: 
    
   # draw a rectangle in a face 
   cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) 
   cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
   roi_gray = gray[y:y+h, x:x+w] 
   roi_color = img[y:y+h, x:x+w] 
  
   # detecting smile within the face roi 
   mouths = mouth_cascade.detectMultiScale(roi_gray, 1.5, 20) 
   if len(mouths) > 0: 
      print("mouth detected") 
      for (sx, sy, sw, sh) in mouths: 
         cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
         cv2.putText(roi_color, "mouth", (sx, sy), 
         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
   else: 
      print("mouth not detected") 
 
# Display an image in a window 
cv2.imshow('Mouth Image',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
