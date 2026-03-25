from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
from PIL import Image
import io
import tensorflow as tf
import cv2
import os
import subprocess

app = Flask(__name__)

model = tf.keras.models.load_model("model.keras")

# garantir pasta feedback
os.makedirs("feedback", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

# 🔥 pré-processamento centralizado
def process_image(image):
    image = np.array(image)
    image = 255 - image

    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # engrossar traço
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # encontrar número
    coords = np.column_stack(np.where(image > 0))

    if coords.size > 0:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        image = image[x_min:x_max+1, y_min:y_max+1]

    image = cv2.resize(image, (20, 20))
    image = cv2.GaussianBlur(image, (3,3), 0)

    new_image = np.zeros((28, 28), dtype=np.float32)

    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2

    new_image[x_offset:x_offset+20, y_offset:y_offset+20] = image

    new_image = new_image / 255.0

    return new_image.reshape(1, 28, 28, 1)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")

    processed = process_image(image)

    prediction = model.predict(processed)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": digit,
        "confidence": confidence
    })

# 🔥 salvar feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json

    image_data = base64.b64decode(data["image"].split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")

    processed = process_image(image)

    label = data["label"]

    np.save(f"feedback/img_{label}_{np.random.randint(100000)}.npy", processed[0])

    return jsonify({"status": "salvo"})

# 🔥 re-treinar modelo
@app.route("/retrain", methods=["POST"])
def retrain():
    subprocess.run(["python", "train_model.py"])

    global model
    model = tf.keras.models.load_model("model.keras")

    return jsonify({"status": "modelo atualizado"})

if __name__ == "__main__":
    app.run(debug=True)