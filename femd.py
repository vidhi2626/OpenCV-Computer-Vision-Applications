import cv2

def detect_feature(image_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    mouth_cascade = cv2.CascadeClassifier('mouth.xml')
    nose_cascade = cv2.CascadeClassifier('nose.xml')  # Assuming you have a nose cascade xml

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error: Could not read image at {image_path}')
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 30))
    print(f'Found {len(faces)} face(s)')

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Region of interest (ROI) for eyes, mouth, and nose
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Detect mouths
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=25, minSize=(30, 30))
        for (my, mx, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        # Detect noses
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)

    # Display the result
    cv2.imshow('Face, Eyes, Mouth, and Nose Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the image path
image_path = 'woman3.jpg'
detect_feature(image_path)
