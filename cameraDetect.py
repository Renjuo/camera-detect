import cv2

# Load the pre-trained Haar Cascade Classifiers for face and eyes detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open the camera
cam = cv2.VideoCapture(0)

# Create a display window to show the results
cv2.namedWindow("test")

img_counter = 0

while True:
    # Read a frame from the camera
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Region of Interest (ROI) for eyes within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("test", frame)

    # Wait for a key press
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed, exit the loop
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed, save the frame as an image
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

# Release the camera and close the display window
cam.release()
cv2.destroyAllWindows()
