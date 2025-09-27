import cv2
import numpy as np

def cartoonize(img):
    # Apply bilateral filter and edge detection for cartoon effect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def glitch_effect(img):
    # Simple glitch: shift color channels
    b, g, r = cv2.split(img)
    rows, cols = b.shape
    shift = np.random.randint(1, 10)
    b = np.roll(b, shift, axis=1)
    g = np.roll(g, -shift, axis=0)
    r = np.roll(r, shift, axis=0)
    return cv2.merge((b, g, r))

def sepia_effect(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia

def emboss_effect(img):
    kernel = np.array([[ -2, -1, 0],
                       [ -1,  1, 1],
                       [  0,  1, 2]])
    embossed = cv2.filter2D(img, -1, kernel) + 128
    return np.clip(embossed, 0, 255).astype(np.uint8)

def sobel_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    return sobel_bgr

def red_tint(img):
    tinted = img.copy()
    tinted[..., 2] = np.clip(tinted[..., 2] + 80, 0, 255)
    return tinted

def pixelate(img, pixel_size=16):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def sharpen_effect(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def blue_tint(img):
    tinted = img.copy()
    tinted[..., 0] = np.clip(tinted[..., 0] + 80, 0, 255)
    return tinted

def put_label(img, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    shadow = (0, 0, 0)
    org = (10, 30)
    img = img.copy()
    cv2.putText(img, label, (org[0]+2, org[1]+2), font, font_scale, shadow, thickness+2, cv2.LINE_AA)
    cv2.putText(img, label, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read one frame to get shape
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        cap.release()
        return

    height, width, channels = frame.shape
    # Initialize per-pixel cumulative sum and count
    cumulative_sum = np.zeros((height, width, channels), dtype=np.float64)
    pixel_count = np.zeros((height, width, 1), dtype=np.int32)
    max_frames = 100  # Maximum number of frames to consider in the average

    ghost_trail = False
    edge_overlay = True  # Always on

    colormaps = [
        cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_COOL,
        cv2.COLORMAP_HSV, cv2.COLORMAP_OCEAN, cv2.COLORMAP_PINK
    ]
    cmap_idx = 0
    freeze_average = False
    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Mirror flip the camera
        frame = cv2.flip(frame, 1)

        if not freeze_average:
            # Update cumulative sum and count for each pixel
            cumulative_sum += frame
            pixel_count += 1

            # Cap the pixel_count at max_frames
            pixel_count = np.minimum(pixel_count, max_frames)

            # If pixel_count exceeds max_frames, subtract the oldest frame's value
            # For simplicity, use exponential moving average after max_frames
            if np.any(pixel_count == max_frames):
                alpha = 1.0 / max_frames
                cumulative_sum = (1 - alpha) * cumulative_sum + alpha * frame

        # Compute per-pixel average
        average = (cumulative_sum / pixel_count).astype(np.uint8)

        # Show the absolute difference (motion/change map)
        diff = cv2.absdiff(frame, average)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_color = cv2.applyColorMap(diff_gray, colormaps[cmap_idx])

        # Ghost trail effect: blend current frame and average
        if ghost_trail:
            ghost = cv2.addWeighted(frame, 0.5, average, 0.5, 0)
        else:
            ghost = average

        # Edge overlay: overlay Canny edges from average onto the live frame (always on)
        edges = cv2.Canny(average, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay = frame.copy()
        overlay[edges > 0] = (0, 255, 0)  # Green edges
        edge_result = overlay

        # --- Additional Fun Effects ---

        # Inverted colors
        inverted = cv2.bitwise_not(frame)

        # Posterization
        div = 64
        poster = frame // div * div + div // 2

        # HSV cycling (rotate hue)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[...,0] = (hsv[...,0] + (cv2.getTickCount() // 1000000) % 180) % 180
        hsv_cycle = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Glitch effect
        glitch = glitch_effect(frame)

        # Cartoon effect
        cartoon = cartoonize(frame)

        # Sepia effect
        sepia = sepia_effect(frame)

        # Emboss effect
        emboss = emboss_effect(frame)

        # Sobel edge detection
        sobel = sobel_edge(frame)

        # Red tint
        red = red_tint(frame)

        # Pixelate
        pixelated = pixelate(frame, pixel_size=16)

        # New effects for the fourth row
        cartoon2 = cartoonize(frame)
        sharpen = sharpen_effect(frame)
        blue = blue_tint(frame)

        # Resize all images to the same size for stacking
        grid_h, grid_w = 180, 240
        edge_resized = cv2.resize(edge_result, (grid_w, grid_h))
        ghost_resized = cv2.resize(ghost, (grid_w, grid_h))
        diff_resized = cv2.resize(diff_color, (grid_w, grid_h))
        inverted_resized = cv2.resize(inverted, (grid_w, grid_h))
        glitch_resized = cv2.resize(glitch, (grid_w, grid_h))
        sepia_resized = cv2.resize(sepia, (grid_w, grid_h))
        emboss_resized = cv2.resize(emboss, (grid_w, grid_h))
        sobel_resized = cv2.resize(sobel, (grid_w, grid_h))
        blue_resized = cv2.resize(blue, (grid_w, grid_h))

        # Add labels to each effect
        edge_resized = put_label(edge_resized, "Edge Overlay")
        ghost_resized = put_label(ghost_resized, "Ghost/Average")
        diff_resized = put_label(diff_resized, "Motion Heatmap")
        inverted_resized = put_label(inverted_resized, "Inverted")
        glitch_resized = put_label(glitch_resized, "Glitch")
        sepia_resized = put_label(sepia_resized, "Sepia")
        emboss_resized = put_label(emboss_resized, "Emboss")
        sobel_resized = put_label(sobel_resized, "Sobel Edge")
        blue_resized = put_label(blue_resized, "Blue Tint")

        # 3x3 grid (9 effects, fill with pixelate and red tint for completeness)
        pixel_resized = cv2.resize(pixelated, (grid_w, grid_h))
        red_resized = cv2.resize(red, (grid_w, grid_h))
        pixel_resized = put_label(pixel_resized, "Pixelate")
        red_resized = put_label(red_resized, "Red Tint")

        row1 = np.hstack((edge_resized, ghost_resized, diff_resized))
        row2 = np.hstack((inverted_resized, glitch_resized, sepia_resized))
        row3 = np.hstack((emboss_resized, sobel_resized, blue_resized))

        grid = np.vstack((row1, row2, row3))

        cv2.imshow('Fun Effects Grid', grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            ghost_trail = not ghost_trail
        elif key == ord('c'):
            cmap_idx = (cmap_idx + 1) % len(colormaps)
        elif key == ord('f'):
            freeze_average = not freeze_average
        elif key == ord('s'):
            cv2.imwrite(f"snapshot_{snapshot_count}.png", grid)
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
