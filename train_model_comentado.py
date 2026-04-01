# Importa o TensorFlow, que é a biblioteca principal para criar redes neurais
import tensorflow as tf

# Importa camadas (layers) e modelos prontos do Keras (API do TensorFlow)
from tensorflow.keras import layers, models

# Importa o dataset MNIST (números escritos à mão)
from tensorflow.keras.datasets import mnist

# Importa ferramenta para criar variações das imagens (augmentation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Biblioteca para manipulação de arrays (matrizes numéricas)
import numpy as np

# Biblioteca para trabalhar com arquivos e diretórios
import os


# ============================================================
# 📦 1. CARREGAR OS DADOS
# ============================================================

# Aqui carregamos o MNIST, que já vem separado em:
# - treino (x_train, y_train)
# - teste (x_test, y_test)
# x = imagens | y = rótulos (número correto)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ============================================================
# 🎯 2. NORMALIZAÇÃO DOS DADOS
# ============================================================

# As imagens vão de 0 a 255 (escala de cinza)
# Dividimos por 255 para deixar entre 0 e 1
# Isso ajuda a rede neural a aprender melhor

x_train = x_train / 255.0
x_test = x_test / 255.0


# ============================================================
# 🧠 3. AJUSTAR FORMATO PARA CNN
# ============================================================

# A CNN espera dados no formato:
# (quantidade, altura, largura, canais)

# -1 = o Python calcula automaticamente a quantidade de imagens
# 28x28 = tamanho da imagem
# 1 = canal (preto e branco)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# ============================================================
# 🔁 4. CARREGAR DADOS DE FEEDBACK DOS ALUNOS
# ============================================================

# Se existir uma pasta chamada "feedback" 🔥
# significa que já temos dados novos coletados pelos usuários

if os.path.exists("feedback"):

    X_new = []  # imagens novas
    y_new = []  # rótulos corretos informados pelos alunos

    # percorre todos os arquivos da pasta
    for file in os.listdir("feedback"):

        # carrega a imagem salva (array numpy)
        data = np.load(f"feedback/{file}")

        # extrai o número correto do nome do arquivo
        # exemplo: img_7_1234.npy → número = 7
        label = int(file.split("_")[1])

        X_new.append(data)
        y_new.append(label)

    # se tiver dados novos
    if len(X_new) > 0:

        # transforma listas em arrays numpy
        X_new = np.array(X_new)
        y_new = np.array(y_new)

        # junta os dados novos com os dados originais
        x_train = np.concatenate((x_train, X_new))
        y_train = np.concatenate((y_train, y_new))

        print(f"📊 Adicionados {len(X_new)} novos exemplos!")

    
        # "A IA está aprendendo com vocês agora!" 💬


# ============================================================
# 🔄 5. DATA AUGMENTATION (VARIAÇÕES DAS IMAGENS)
# ============================================================

# Aqui criamos pequenas variações nas imagens para melhorar o aprendizado
# Isso simula diferentes formas de escrita dos alunos
datagen = ImageDataGenerator(

    # gira levemente a imagem
    rotation_range=10,

    # zoom leve
    zoom_range=0.1,

    # desloca horizontalmente
    width_shift_range=0.1,

    # desloca verticalmente
    height_shift_range=0.1
)

# "Ensina" o gerador como são os dados
datagen.fit(x_train)


# ============================================================
# 🧠 6. CRIAÇÃO DO MODELO (REDE NEURAL CNN)
# ============================================================

model = models.Sequential([

    # define o formato de entrada
    layers.Input(shape=(28, 28, 1)),

    # 🔍 camada convolucional (detecta padrões como linhas e curvas)
    layers.Conv2D(32, (3,3), activation='relu'),

    # reduz o tamanho da imagem mantendo características importantes
    layers.MaxPooling2D(2,2),

    # segunda camada convolucional (detecta padrões mais complexos)
    layers.Conv2D(64, (3,3), activation='relu'),

    layers.MaxPooling2D(2,2),

    # transforma matriz em vetor (prepara para classificação)
    layers.Flatten(),

    # camada densa (aprende combinações de padrões)
    layers.Dense(128, activation='relu'),

    # camada final com 10 saídas (0 a 9)
    layers.Dense(10, activation='softmax')
])


# ============================================================
# ⚙️ 7. CONFIGURAÇÃO DO MODELO
# ============================================================
model.compile(

    # algoritmo de aprendizado
    optimizer='adam',

    # função de erro (compara previsão com resposta correta)
    loss='sparse_categorical_crossentropy',

    # métrica de desempenho
    metrics=['accuracy']
)


# ============================================================
# 🚀 8. TREINAMENTO DO MODELO
# ============================================================

# usamos o datagen (com variações) para treinar
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10
)


# "Aqui a IA está aprendendo observando milhares de exemplos"💬


# ============================================================
# 🧪 9. AVALIAÇÃO DO MODELO
# ============================================================

# testa o modelo com dados que ele nunca viu
loss, acc = model.evaluate(x_test, y_test)

print("Acurácia:", acc)


# "Isso mostra o quão bem a IA generaliza" 💬


# ============================================================
# 💾 10. SALVAR O MODELO
# ============================================================

# salva o modelo treinado para uso no sistema Flask
model.save("model.keras")

print("✅ Modelo atualizado!")

# "Agora podemos usar essa IA sem precisar treinar de novo" 💬

"""
👉 “A IA não vê números, ela vê pixels”
👉 “Treinar é mostrar exemplos repetidamente”
👉 “Data augmentation simula diferentes pessoas escrevendo”
👉 “A IA melhora quando recebe novos dados”
"""