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


Technologies Used
Python: The core programming language for the entire application.
Flask: A lightweight web framework used to build both the web interface (serving HTML/CSS) and the prediction API endpoints.
TensorFlow / Keras: For loading and running the pre-trained VGG16 deep learning model.
Numpy: Essential for numerical operations, especially image manipulation.
OpenCV (cv2): Used for robust image preprocessing (reading, resizing) and generating/overlaying Grad-CAM heatmaps.
HTML5 & CSS3: For structuring and styling the responsive web frontend.
Installation and Setup (Local Development)
To get NeuroScan up and running on your local machine, follow these steps:
1. Clone the Repository
code
Bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
2. Set up Python Environment
Create and activate a virtual environment (highly recommended for dependency management):
code
Bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install the required Python packages:
code
Bash
pip install Flask tensorflow numpy opencv-python-headless
(Note: opencv-python-headless is used for server environments to avoid GUI dependencies.)
3. Place Your Trained Model
Ensure your trained Keras model file, brain_tumor_detection_vgg16.h5, is located in the root directory of the project, next to app.py. If it's elsewhere, update the MODEL_PATH in app.py.
4. Create static and templates Folders
Make sure you have the static and templates folders as described in the Folder Structure section, and place index.html, mri_hero.jpg, and upload.png in their respective locations.
5. Run the Flask Application
From the root directory of your project, run the Flask application:
code
Bash
python app.py
The application will typically start on http://127.0.0.1:5000 (or localhost:5000).
Usage
Start the Application: Run python app.py and navigate to http://127.0.0.1:5000 in your web browser.
Upload MRI Scan: Click the "Click to browse MRI image" area (or the hidden file input it triggers) and select an MRI image file (JPG or PNG) from your computer.
Analyze Scan: Click the "Analyze Scan" button.
View Results: The application will display the prediction ("Tumor Detected" or "No Tumor Detected"), the confidence score, the original uploaded image, and the Grad-CAM heatmap overlay.
Grad-CAM Explanation
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used to make CNN models more interpretable. It produces a coarse localization map highlighting the important regions in the input image that were used by the model to make its prediction.
In this application:
When an MRI image is uploaded, the model not only predicts the presence of a tumor but also generates a Grad-CAM heatmap.
This heatmap is then overlaid onto the original MRI image, where warmer colors (e.g., red, yellow) indicate areas of high importance for the model's decision, and cooler colors (e.g., blue) indicate less important areas.
This allows users, particularly medical professionals, to visually confirm if the model is focusing on relevant anatomical regions or potential tumor areas, thereby increasing trust in the AI's predictions.
Model Training Details
The deep learning model is built upon the VGG16 architecture, a powerful CNN known for its effectiveness in image classification tasks. It was initialized with weights pre-trained on the vast ImageNet dataset (transfer learning) and then fine-tuned specifically for brain tumor detection.
Base Model: VGG16 (pre-trained on ImageNet, include_top=False)
Custom Head: Added Flatten, Dense (256 units, ReLU), Dropout (0.5), and final Dense (1 unit, Sigmoid) layers for binary classification.
Input Image Size: All images are resized to 224x224 pixels and normalized to [0, 1].
Data Augmentation: Techniques like rotation, width/height shifts, shear, zoom, horizontal/vertical flips, and brightness adjustments were applied during training to enhance the model's robustness and ability to generalize to unseen data.
Optimizer: Adam with a learning rate of 1e-4 (initially) or 1e-5 (for fine-tuning).
Loss Function: Binary Crossentropy, suitable for binary classification tasks.
Metrics: Accuracy, Validation Accuracy, and further evaluated with Precision, Recall, F1-Score, Confusion Matrix, and ROC AUC.
Class Weighting: Implemented to address potential class imbalance in the training dataset.
License
This project is licensed under the MIT License - see the LICENSE file for details.
