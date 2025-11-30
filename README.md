# Brain Tumor Detection – NeuroScan Web Application

## Project Overview

This project presents **NeuroScan**, a web-based application designed for the detection of brain tumors in MRI images. The application's backend and frontend are powered by a Flask server, integrating a fine-tuned VGG16 Convolutional Neural Network (CNN) for robust image analysis. A key feature is the inclusion of **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which generates a heatmap overlay on the MRI images, visually highlighting the regions the model focuses on for its prediction. This enhances trust and interpretability for medical professionals.

Users can upload an MRI image through a clean interface, receive an immediate prediction (Tumorous / No Tumor) with a confidence score, and visually inspect the Grad-CAM heatmap to understand the model's decision-making process.

## Features

*   **Image Upload:** Seamlessly upload MRI images (JPG/PNG formats).
*   **Real-time Prediction:** Get instant deep learning-based predictions.
*   **Confidence Scoring:** Provides a percentage confidence for the model's classification.
*   **Grad-CAM Heatmaps:** Visualize the critical regions in the MRI scan that influenced the tumor detection, increasing model interpretability.
*   **User-Friendly Interface:** A clean and intuitive design for ease of use.
*   **Transfer Learning (VGG16):** Utilizes the powerful VGG16 architecture, pre-trained on ImageNet, fine-tuned for this specific medical imaging task.
*   **Data Augmentation:** The underlying model was trained with extensive data augmentation to improve generalization and robustness.

## Demo

Here's an example of NeuroScan in action:
`![Alt text](image.png)
