import numpy as np
import cv2

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device")
else:
    while(True):
        # reading the frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        # displaying the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # breaking the loop if the user types q
            # note that the video window must be highlighted!
            break

cap.release()
cv2.destroyAllWindows()
# the following is necessary on the mac,
# maybe not on other platforms:
cv2.waitKey(1)
