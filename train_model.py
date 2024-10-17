import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import ModelCheckpoint  # Import the ModelCheckpoint
import json

# Function to load and preprocess images
def load_images_from_folder(folder, img_size=(224, 224)):  # Updated to (224, 224)
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)  # Resize to the new dimensions
                images.append(img_resized)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names

# Load and preprocess the data
data_folder = "dataset"
images, labels, class_names = load_images_from_folder(data_folder)
print(f"Loaded {len(images)} images for training across {len(class_names)} classes.")

if len(images) == 0:
    raise ValueError("No images found. Please check your dataset folder.")

images = images.reshape(-1, 224, 224, 1) / 255.0  # Normalize images (adjusted for new size)

# Save class names to a JSON file
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved to class_names.json")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),  # Adjust input shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Model checkpoint
model_checkpoint = ModelCheckpoint("best_hand_sign_model.keras", save_best_only=True, monitor='val_accuracy')

# Train the model
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[model_checkpoint])

# Save the trained model
cnn_model.save("hand_sign_cnn_model.keras")  # Changed to .keras format
print("Model saved as hand_sign_cnn_model.keras")

# Evaluate the model
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
