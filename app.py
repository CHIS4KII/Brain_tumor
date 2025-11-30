from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

model = load_model("brain_tumor_detection_vgg16.h5")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_gradcam(model, img_array, last_conv_layer="block5_conv3"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = np.zeros(shape=conv_outputs.shape[:2])
    for i in range(conv_outputs.shape[-1]):
        heatmap += conv_outputs[:, :, i] * pooled_grads[i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap, (224, 224))
    return heatmap


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    confidence = None
    overlay_image = None
    uploaded_image = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            file_path = os.path.join("static", file.filename).replace("\\", "/")
            file.save(file_path)
            uploaded_image = file_path

            img_array = preprocess_image(file_path)
            result = float(model.predict(img_array)[0][0])

            if result > 0.5:
                prediction_text = "⚠️ Tumor Detected"
                confidence = round(result * 100, 2)
            else:
                prediction_text = "✔️ No Tumor Detected"
                confidence = round((1 - result) * 100, 2)

            heatmap = generate_gradcam(model, img_array)
            original = cv2.imread(file_path)
            original = cv2.resize(original, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)

            overlay_path = "static/overlay_result.jpg"
            cv2.imwrite(overlay_path, overlay)
            overlay_image = overlay_path

    return render_template(
        "index.html",
        prediction=prediction_text,
        confidence=confidence,
        image=uploaded_image,
        overlay_image=overlay_image
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return {"error": "No file uploaded"}

    file = request.files["image"]
    file_path = os.path.join("static", file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    result = float(model.predict(img)[0][0])

    return {
        "prediction": "tumor" if result > 0.5 else "no_tumor",
        "confidence": round(result if result > 0.5 else 1 - result, 4)
    }


if __name__ == "__main__":
    app.run(debug=True)
