# 🌱 AI-Based Crop Disease Detection

## Project Description

This project is an AI-based application that detects diseases in tomato leaves from uploaded images.

The application uses a trained deep learning model to classify tomato leaves into four categories:

- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Healthy

## Objective

The objective of this project is to use Artificial Intelligence and Deep Learning for detecting diseases in tomato leaves.

## Technologies Used

- Python
- TensorFlow
- MobileNetV2
- Streamlit
- NumPy
- Pillow

## Features

- Upload tomato leaf images
- Upload multiple images at the same time
- Predict the condition of the tomato leaf
- Display the predicted disease
- Display prediction confidence

## How the Model Works

1. The dataset is loaded using TensorFlow.
2. The images are resized to 224 × 224 pixels.
3. MobileNetV2 is used as the pretrained base model.
4. The pretrained layers are frozen.
5. Additional layers are added for classification.
6. The model is trained for 5 epochs.
7. The trained model is saved as `crop_disease_model.keras`.
8. The Streamlit application loads the trained model.
9. The uploaded image is resized to 224 × 224 pixels.
10. The model predicts the class of the leaf.
11. The predicted disease and confidence are displayed.

## Classes

The model classifies images into:

1. Early Blight
2. Late Blight
3. Leaf Mold
4. Healthy

## Project Files

```text
crop-disease-detection/
│
├── app.py
├── predict.py
├── train_model.py
├── crop_disease_model.keras
├── requirements.txt
├── README.md
└── .gitignore