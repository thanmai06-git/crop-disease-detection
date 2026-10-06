import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("crop_disease_model.keras")

# Class names
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

# Page settings
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱"
)

# Title
st.title("🌱 AI-Based Crop Disease Detection")

st.write(
    "Upload tomato leaf images and the AI model will "
    "predict whether the leaf is healthy or affected by a disease."
)

st.divider()

# Upload multiple images
uploaded_files = st.file_uploader(
    "📷 Upload tomato leaf images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Process uploaded images
if uploaded_files:

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch"
        )

        # Resize for the trained model
        image = image.resize((224, 224))

        # Convert image to array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        predicted_class = np.argmax(prediction[0])

        confidence = prediction[0][predicted_class] * 100

        # Get disease name
        disease_name = class_names[predicted_class]
        disease_name = disease_name.replace("Tomato___", "")
        disease_name = disease_name.replace("_", " ")

        st.subheader("🔍 Prediction Result")

        st.success(
            f"Predicted Condition: {disease_name}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.divider()