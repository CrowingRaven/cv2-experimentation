import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert each pixel to the cosine of itself
    # Normalize pixel values to range [0, 1] for cosine calculation
    normalized = gray / 255.0
    cosine_transformed = np.cos(normalized * (np.pi*2))  # Scale to [0, π] for better variation

    # Scale back to [0, 255] for display
    output = (cosine_transformed * 255).astype(np.uint8)

    # Display the resulting frame
    cv2.imshow('Cosine Transformed', output)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()