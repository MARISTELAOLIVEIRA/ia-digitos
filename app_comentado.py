# ============================================================
# 🌐 1. IMPORTAÇÕES
# ============================================================

# Flask: framework web (cria o servidor)
from flask import Flask, render_template, request, jsonify

# numpy: manipulação de matrizes (imagens são matrizes de pixels)
import numpy as np

# base64: formato usado para enviar a imagem do navegador para o backend
import base64

# PIL: biblioteca para manipular imagens
from PIL import Image

# io: permite tratar bytes como arquivos
import io

# TensorFlow: onde está nosso modelo de IA
import tensorflow as tf

# OpenCV: processamento de imagem (muito importante!)
import cv2

# os: manipulação de arquivos e diretórios
import os

# subprocess: permite rodar outro script (treinar novamente a IA)
import subprocess


# ============================================================
# 🚀 2. INICIANDO A APLICAÇÃO
# ============================================================

# cria a aplicação Flask
app = Flask(__name__)

# carrega o modelo já treinado
model = tf.keras.models.load_model("model.keras")


# ============================================================
# 📁 3. GARANTIR PASTA DE FEEDBACK
# ============================================================

# cria a pasta "feedback" se ela não existir
# aqui vamos salvar os erros corrigidos pelos alunos
os.makedirs("feedback", exist_ok=True)


# ============================================================
# 🏠 4. ROTA PRINCIPAL
# ============================================================

@app.route("/")
def index():
    # renderiza a página HTML
    return render_template("index.html")


# ============================================================
# 🧠 5. PRÉ-PROCESSAMENTO DA IMAGEM (PARTE MAIS IMPORTANTE)
# ============================================================

def process_image(image):

    # transforma imagem em matriz numpy
    image = np.array(image)

    # 🔥 inverte cores (fundo preto, número branco)
    image = 255 - image

    # 🔥 binarização: remove ruídos
    # tudo abaixo de 50 vira preto, acima vira branco
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)


    # 🔥 engrossar traço (ajuda a IA a reconhecer melhor)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)


    # 🔍 encontrar onde está o número na imagem
    coords = np.column_stack(np.where(image > 0))

    if coords.size > 0:

        # pega limites do número
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        # recorta apenas a área do número
        image = image[x_min:x_max+1, y_min:y_max+1]


    # 🔄 redimensiona para 20x20
    image = cv2.resize(image, (20, 20))

    # 🔥 suaviza (remove bordas duras)
    image = cv2.GaussianBlur(image, (3,3), 0)


    # 🧱 cria imagem 28x28 (padrão MNIST)
    new_image = np.zeros((28, 28), dtype=np.float32)

    # calcula posição para centralizar
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2

    # coloca o número no centro
    new_image[x_offset:x_offset+20, y_offset:y_offset+20] = image


    # 📉 normaliza (0 a 1)
    new_image = new_image / 255.0


    # 🧠 formato final esperado pela CNN
    return new_image.reshape(1, 28, 28, 1)


# ============================================================
# 🤖 6. ROTA DE PREVISÃO (IA)
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # recebe imagem enviada pelo frontend
    data = request.json["image"]

    # remove cabeçalho base64 e decodifica
    image_data = base64.b64decode(data.split(",")[1])

    # transforma em imagem
    image = Image.open(io.BytesIO(image_data)).convert("L")

    # processa imagem
    processed = process_image(image)

    # faz previsão com a IA
    prediction = model.predict(processed)

    # pega o número com maior probabilidade
    digit = int(np.argmax(prediction))

    # pega a confiança da previsão
    confidence = float(np.max(prediction))


    # retorna resultado para o frontend
    return jsonify({
        "prediction": digit,
        "confidence": confidence
    })


# ============================================================
# 💬 7. ROTA DE FEEDBACK (APRENDIZADO HUMANO)
# ============================================================

@app.route("/feedback", methods=["POST"])
def feedback():

    data = request.json

    # decodifica imagem
    image_data = base64.b64decode(data["image"].split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")

    # aplica o mesmo pré-processamento
    processed = process_image(image)

    # pega o número correto informado pelo usuário
    label = data["label"]

    # salva imagem + label para treino futuro
    np.save(
        f"feedback/img_{label}_{np.random.randint(100000)}.npy",
        processed[0]
    )

    return jsonify({"status": "salvo"})


# ============================================================
# 🔁 8. RE-TREINAR A IA
# ============================================================

@app.route("/retrain", methods=["POST"])
def retrain():

    # executa o script de treino novamente
    subprocess.run(["python", "train_model.py"])

    # recarrega o modelo atualizado
    global model
    model = tf.keras.models.load_model("model.keras")

    return jsonify({"status": "modelo atualizado"})


# ============================================================
# ▶️ 9. INICIAR SERVIDOR
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
    
"""
👉 “A IA não vê o desenho, ela vê números (pixels)”
👉 “Antes de usar IA, precisamos preparar os dados”
👉 “Vocês estão treinando a IA com feedback”
👉 “IA melhora com dados, não com mágica”
👉 “Treinar é mostrar exemplos repetidamente”
👉 “Aqui a IA está aprendendo observando milhares de exemplos”💬
"""