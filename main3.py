import cv2
import numpy as np

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize variables for pixel averaging
    total_pixels = 0
    cumulative_sum = np.array([0, 0, 0], dtype=np.float64)  # For BGR channels

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process each pixel
        height, width, _ = frame.shape
        for y in range(height):
            for x in range(width):
                # Get the current pixel's color
                current_pixel = frame[y, x]

                # Update cumulative sum and total pixel count
                cumulative_sum += current_pixel
                total_pixels += 1

                # Compute the average color
                average_color = (cumulative_sum / total_pixels).astype(np.uint8)

                # Set the pixel to the average color
                frame[y, x] = average_color

        # Display the resulting frame
        cv2.imshow('Averaged Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()