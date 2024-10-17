import os
import cv2
import numpy as np

# Function to capture images for a specific gesture
def capture_images(gesture_name, num_samples=200):
    # Create the directory for the gesture if it doesn't exist
    if not os.path.exists(f"dataset/{gesture_name}"):
        os.makedirs(f"dataset/{gesture_name}")

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define a Region of Interest (ROI) for hand placement
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # Show instructions on the screen
        cv2.putText(frame, f"Collecting {gesture_name} images: {count}/{num_samples}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the grayscale image within the ROI
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Augmentation: Randomly adjust brightness
        brightness_factor = np.random.uniform(0.5, 1.5)
        roi_bright = cv2.convertScaleAbs(roi_gray, alpha=brightness_factor)

        img_path = f"dataset/{gesture_name}/{gesture_name}_{count}.jpg"
        cv2.imwrite(img_path, roi_bright)

        # Display the augmented image for preview
        cv2.imshow("Augmented Image Preview", roi_bright)

        count += 1

        # Stop collecting after reaching the required number of samples
        if count >= num_samples:
            break

        # Press 'q' to quit the capturing early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting {num_samples} images for {gesture_name}.")

# List of gestures to collect
gestures = ['hi', 'hello', 'love_you']

# Collect data for each gesture
for gesture in gestures:
    print(f"Starting to collect data for {gesture}. Press 'q' to stop early.")
    capture_images(gesture_name=gesture, num_samples=200)  # Adjust the num_samples as needed
