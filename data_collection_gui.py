import cv2
import os
import tkinter as tk
from tkinter import messagebox

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the class labels
class_names = ["hi", "hello", "love you"]

# Specify the folder where you want to save images
save_folder = "collected_data"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Set the number of images you want to capture per class
num_images_per_class = 100
image_count = {"hi": 0, "hello": 0, "love you": 0}

# Tkinter window setup
root = tk.Tk()
root.title("Hand Sign Dataset Collection")

# Instructions label
instructions_label = tk.Label(root, text="Press the buttons to capture images for each class", font=("Arial", 14))
instructions_label.pack(pady=10)

# Status label to show which class is being captured
status_label = tk.Label(root, text="", font=("Arial", 12))
status_label.pack(pady=10)


# Capture images function
def collect_images(class_label):
    global image_count
    count = image_count[class_label]

    if count >= num_images_per_class:
        messagebox.showinfo("Completed", f"Collected {num_images_per_class} images for '{class_label}'")
        return

    # Collect images for the current class
    status_label.config(text=f"Collecting images for: {class_label}")

    while count < num_images_per_class:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        # Display the frame
        cv2.imshow("Image Capture", frame)

        # Save the image when 'c' is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            img_name = f"{class_label}_{count}.jpg"
            img_path = os.path.join(save_folder, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1
            image_count[class_label] = count

        # Press 'q' to quit the collection process early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    status_label.config(text=f"Completed collecting for '{class_label}'")


# Create buttons for each class
for class_name in class_names:
    button = tk.Button(root, text=f"Capture {class_name.upper()} Images", font=("Arial", 12),
                       command=lambda c=class_name: collect_images(c))
    button.pack(pady=5)

# Start the Tkinter loop
root.mainloop()

# Release the webcam when the window is closed
cap.release()
cv2.destroyAllWindows()
