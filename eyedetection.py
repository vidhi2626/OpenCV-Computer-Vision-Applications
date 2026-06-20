import cv2
img=cv2.imread('womanwithglass.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
faces=face_cascade.detectMultiScale(gray,1.3,4)
print('Number of detectd faces:' ,len(faces))
for(x,y,w,h) in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
    eyes=eye_cascade.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh)in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,255),2)
cv2.imshow('Eyes detection',img)
cv2.waitKey(0)
cv2.destoryAllWindow()
