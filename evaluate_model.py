import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the trained model
model = load_model("hand_sign_cnn_model.h5")
print("Model loaded successfully.")

# Load class names from JSON
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Define the dataset directory
dataset_dir = "your_dataset_directory"  # Replace with your dataset directory
images = []
labels = []

# Load images and labels
for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    if os.path.isdir(label_dir):
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (64, 64))
            images.append(img_resized)
            labels.append(class_names.index(label))  # Assuming class_names corresponds to the label

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images.astype('float32') / 255.0
images = np.reshape(images, (-1, 64, 64, 1))  # Reshape to match model input

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the unique classes in the test set
unique_classes = np.unique(y_test)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=[class_names[i] for i in unique_classes], labels=unique_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))
