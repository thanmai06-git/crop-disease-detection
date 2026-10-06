import tensorflow as tf
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("crop_disease_model.keras")

# Class names
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

# Get images from test_images
image_files = os.listdir("test_images")

# Select the first image
image_path = os.path.join("test_images", image_files[0])

print("Testing image:", image_path)

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_array)

# Find the class with highest probability
predicted_class = np.argmax(prediction[0])

# Get confidence
confidence = prediction[0][predicted_class] * 100

print("Predicted disease:", class_names[predicted_class])
print("Confidence:", round(confidence, 2), "%")