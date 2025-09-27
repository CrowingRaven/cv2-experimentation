import numpy as np
import cv2
import math

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Camera is already in use or not accessible.")
else:
    ret, prev_frame = cam.read()  # Capture the first frame for subtraction
    if not ret:
        print("Error: Unable to read from camera.")
    else:
        while True:
            ret, curr_frame = cam.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Perform image subtraction
            diff = cv2.absdiff(curr_gray, prev_gray)
            
            diff = cv2.flip(diff, 1)  # Flip the image horizontally

            # Display the result
            cv2.imshow("Image Subtraction", diff)

            # Update the previous frame
            prev_frame = curr_frame

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cam.release()
cv2.destroyAllWindows()
