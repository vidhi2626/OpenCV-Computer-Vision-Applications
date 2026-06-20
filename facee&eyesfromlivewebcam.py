import cv2
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector=cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture=cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS,5)
while True:
    ret,frame=video_capture.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_detector.detectMultiScale(roi_gray,scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
        for(ex,ey,ew,eh)in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
