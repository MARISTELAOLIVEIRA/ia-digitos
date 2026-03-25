# 🧠 IA que Aprende com Você - Reconhecimento de Dígitos

## 🎓 Aplicação Educacional para Ensino de Deep Learning

Uma aplicação web interativa desenvolvida especialmente para **estudantes de Ciência de Dados** aprenderem na prática os conceitos fundamentais de Deep Learning e Machine Learning. 

Este projeto demonstra de forma didática e visual:
- **Redes Neurais Convolucionais (CNN)** aplicadas a visão computacional
- **Aprendizado Contínuo** (Continuous Learning)
- **Human-in-the-Loop Machine Learning**
- **Processamento e Pré-processamento de Imagens**
- **Ciclo completo de MLOps** (treino, deploy, feedback, retreino)

## 📋 Sobre o Projeto

Esta aplicação demonstra um ciclo completo de Machine Learning com **Human-in-the-Loop**, onde:
- Um modelo de Rede Neural Convolucional (CNN) identifica dígitos desenhados à mão
- Usuários fornecem feedback sobre predições incorretas
- O sistema armazena os feedbacks e retreina o modelo automaticamente
- O modelo evolui e melhora continuamente com novos dados

### 🎯 Objetivos Pedagógicos

**Para Alunos:**
- Visualizar na prática como uma CNN processa e classifica imagens
- Entender o impacto da qualidade dos dados no desempenho do modelo
- Experimentar o conceito de aprendizado contínuo em tempo real
- Compreender métricas de confiança e probabilidade nas predições
- Praticar conceitos de data augmentation e pré-processamento

**Para Professores:**
- Ferramenta interativa para demonstrar conceitos de Deep Learning
- Facilita a compreensão de tópicos abstratos de forma visual
- Permite experimentação hands-on dos alunos
- Base para discussões sobre overfitting, underfitting e generalização
- Exemplo prático de pipeline completo de ML

## ✨ Funcionalidades

- **Desenho Interativo**: Canvas HTML5 para desenhar dígitos de 0-9
- **Predição em Tempo Real**: Reconhecimento instantâneo de dígitos
- **Sistema de Feedback**: Correção de predições incorretas
- **Aprendizado Contínuo**: Re-treinamento do modelo com novos dados
- **Visualização de Confiança**: Exibição da porcentagem de certeza da predição

## 🏗️ Arquitetura

### Tecnologias Utilizadas

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow/Keras
- **Processamento de Imagem**: OpenCV, PIL, NumPy
- **Frontend**: HTML5, JavaScript, Canvas API
- **Dataset Base**: MNIST (70.000 imagens de dígitos manuscritos)

### Modelo de Rede Neural

O modelo utiliza uma arquitetura CNN (Convolutional Neural Network) com:

```
- Camada Conv2D (32 filtros, 3x3) + ReLU
- MaxPooling (2x2)
- Camada Conv2D (64 filtros, 3x3) + ReLU
- MaxPooling (2x2)
- Flatten
- Dense (128 neurônios) + ReLU
- Dense (10 neurônios) + Softmax (saída)
```

## 📂 Estrutura do Projeto

```
Deep/
├── app.py                 # Servidor Flask e endpoints API
├── train_model.py         # Script de treinamento do modelo
├── model.keras            # Modelo treinado salvo
├── requirements.txt       # Dependências do projeto
├── templates/
│   └── index.html        # Interface web
└── feedback/             # Armazenamento de exemplos corrigidos
    └── img_{label}_{id}.npy
```

## 🚀 Instalação e Execução

## 🚀 Instalação e Execução

### ⚡ Quick Start (Usuários Experientes)

```bash
# Clone o projeto
git clone <url-do-repositorio>
cd Deep

# Configure ambiente
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Treine o modelo
python3 train_model.py

# Execute a aplicação
python3 app.py

# Acesse: http://127.0.0.1:5000
```

---

### Pré-requisitos

- Python 3.11 (recomendado)
- pip (gerenciador de pacotes Python)
- VS Code (opcional, mas recomendado)

### 📚 Guia Passo a Passo Completo

#### 1️⃣ Instalar Python 3.11

**No macOS:**
```bash
# Instalar via Homebrew
brew install python@3.11

# Verificar instalação
python3.11 --version
```

**No Windows:**
1. Acesse [python.org/downloads](https://www.python.org/downloads/)
2. Baixe o instalador Python 3.11.x
3. Execute o instalador
4. ✅ **IMPORTANTE**: Marque "Add Python to PATH"
5. Verifique no terminal:
   ```bash
   python --version
   ```

**No Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Verificar instalação
python3.11 --version
```

#### 2️⃣ Clone ou Baixe o Projeto

```bash
# Se usar Git
git clone <url-do-repositorio>
cd Deep

# Ou extraia o arquivo ZIP baixado e navegue até a pasta
cd caminho/para/Deep
```

#### 3️⃣ Criar Ambiente Virtual (venv)

**Por que usar venv?** Isola as dependências do projeto, evitando conflitos com outros projetos Python.

```bash
# Criar ambiente virtual
python3.11 -m venv venv

# Ou no Windows (se python3.11 não funcionar)
python -m venv venv
```

#### 4️⃣ Ativar o Ambiente Virtual

**No macOS/Linux:**
```bash
source venv/bin/activate
```

**No Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

**No Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Confirmação:** Você verá `(venv)` no início da linha do terminal.

#### 5️⃣ Escolher Interpretador no VS Code (Opcional)

Se estiver usando VS Code:

1. Abra a pasta do projeto no VS Code
2. Pressione `Cmd+Shift+P` (macOS) ou `Ctrl+Shift+P` (Windows/Linux)
3. Digite: **"Python: Select Interpreter"**
4. Escolha o interpretador que aparece como: `./venv/bin/python` ou `.\venv\Scripts\python.exe`

**Alternativa rápida:** Clique na versão do Python no canto inferior direito da janela do VS Code.

#### 6️⃣ Instalar Dependências

Com o ambiente virtual ativado:

```bash
# Atualizar pip (recomendado)
pip install --upgrade pip

# Instalar todas as dependências do projeto
pip install -r requirements.txt
```

**Tempo estimado:** 2-5 minutos (depende da conexão de internet)

**Dependências instaladas:**
- `tensorflow==2.15.0` - Framework de Deep Learning
- `numpy==1.26.4` - Computação numérica
- `flask` - Servidor web
- `pillow` - Manipulação de imagens
- `opencv-python` - Processamento de imagens
- `scipy` - Computação científica

#### 7️⃣ Treinar o Modelo Inicial

Se o arquivo `model.keras` não existir, execute:

```bash
python3 train_model.py
```

**O que acontece:**
- Download automático do dataset MNIST (~11 MB)
- Treinamento da CNN por 10 epochs
- Salvamento do modelo treinado em `model.keras`

**Tempo estimado:** 5-10 minutos (primeira execução)

**Saída esperada:**
```
Epoch 1/10
...
Acurácia: 0.98xx
✅ Modelo atualizado!
```

#### 8️⃣ Executar a Aplicação

```bash
python3 app.py
```

**Saída esperada:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

#### 9️⃣ Acessar no Navegador

Abra seu navegador e acesse:
```
http://127.0.0.1:5000
```

ou

```
http://localhost:5000
```

**Pronto!** 🎉 A aplicação está rodando.

#### 🛑 Parar a Aplicação

Pressione `Ctrl+C` no terminal onde a aplicação está rodando.

#### 🔄 Desativar o Ambiente Virtual

Quando terminar de trabalhar:

```bash
deactivate
```

## 🎯 Como Usar

### 1. Desenhar um Dígito
- Use o mouse para desenhar um número (0-9) no canvas branco
- Desenhe de forma clara e centralizada

### 2. Fazer Predição
- Clique no botão **"Prever"**
- O sistema mostrará o dígito reconhecido e a confiança da predição

### 3. Fornecer Feedback

**Se a predição estiver correta:**
- Clique em **"Sim"**

**Se a predição estiver incorreta:**
- Clique em **"Não"**
- Digite o número correto no campo que aparece
- Clique em **"Enviar Correção"**
- O sistema salvará automaticamente na pasta `feedback/`

### 4. Re-treinar o Modelo
- Após coletar vários feedbacks, clique em **"🔁 Re-treinar IA"**
- O modelo será re-treinado incluindo os novos exemplos
- A precisão melhorará com o tempo

### 5. Limpar Canvas
- Clique em **"Limpar"** para desenhar um novo dígito

## 🔄 Fluxo de Aprendizado Contínuo

```
┌─────────────────┐
│  Usuário desenha │
│     dígito       │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Pré-processamento │
│  (28x28, normalização) │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Modelo CNN     │
│  faz predição   │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Exibe resultado │
│  e confiança    │
└────────┬─────────┘
         │
         ▼
    Correto? ────── Não ──┐
         │                 │
        Sim                ▼
         │        ┌─────────────────┐
         │        │ Salva feedback  │
         │        │ em feedback/    │
         │        └────────┬─────────┘
         │                 │
         └─────────────────┤
                           ▼
                  ┌─────────────────┐
                  │  Re-treinar     │
                  │  (manual)       │
                  └─────────────────┘
```

## 🛠️ Detalhes Técnicos

### Pré-processamento de Imagem

O sistema aplica as seguintes transformações:
1. Inversão de cores (fundo preto, traço branco)
2. Binarização com threshold
3. Dilatação para engrossar o traço
4. Detecção e crop do dígito
5. Redimensionamento para 20x20
6. Aplicação de Gaussian Blur
7. Centralização em canvas 28x28
8. Normalização (0-1)

### Endpoints da API

- **GET /** - Página principal
- **POST /predict** - Recebe imagem e retorna predição
- **POST /feedback** - Salva exemplo corrigido
- **POST /retrain** - Re-treina o modelo

### Data Augmentation

Durante o treinamento, o sistema aplica:
- Rotação (±10°)
- Zoom (±10%)
- Deslocamento horizontal e vertical (±10%)

## 📊 Dataset

- **Inicial**: MNIST (60.000 treino + 10.000 teste)
- **Incremental**: Exemplos corrigidos salvos em `feedback/`
- **Formato**: Arrays NumPy 28x28x1 (escala de cinza)

## 🎓 Conceitos Demonstrados

- **Redes Neurais Convolucionais (CNN)**
- **Transfer Learning / Aprendizado Contínuo**
- **Human-in-the-Loop Machine Learning**
- **Data Augmentation**
- **Processamento de Imagens**
- **API REST com Flask**
- **Serialização de Modelos (Keras)**

## 📚 Detalhamento dos Conceitos para Alunos

### 1. Redes Neurais Convolucionais (CNN)
- **Camadas Convolucionais**: Extração automática de features das imagens
- **Pooling (MaxPooling)**: Redução dimensional mantendo informações importantes
- **Flatten**: Transforma matriz 2D em vetor para classificação
- **Camadas Densas**: Classificação final dos dígitos (0-9)

### 2. Pré-processamento de Imagens
- **Normalização**: Escala de pixels de 0-255 para 0-1
- **Binarização**: Separação do dígito do fundo
- **Redimensionamento**: Padronização para 28x28 pixels
- **Filtros**: Gaussian Blur para suavização, dilatação para engrossar traços

### 3. Treinamento e Otimização
- **Função de Perda**: Sparse Categorical Crossentropy (classificação multiclasse)
- **Otimizador**: Adam (adaptive learning rate)
- **Métrica**: Accuracy (acurácia do modelo)
- **Data Augmentation**: Rotação, zoom e shift para aumentar variedade

### 4. Avaliação e Predição
- **Softmax**: Converte saídas em probabilidades (soma = 100%)
- **Confiança**: Probabilidade máxima indica certeza da predição
- **Argmax**: Identifica a classe com maior probabilidade

## 🎯 Exercícios Práticos para Alunos

### Nível Básico
1. **Teste o modelo**: Desenhe todos os dígitos (0-9) e observe as predições
2. **Analise a confiança**: Note quando o modelo tem alta ou baixa certeza
3. **Coleta de dados**: Forneça 10 correções de dígitos mal classificados
4. **Re-treine**: Execute o re-treinamento e compare a performance

### Nível Intermediário
1. **Modifique hiperparâmetros**: Altere epochs, batch_size em `train_model.py`
2. **Ajuste data augmentation**: Mude parâmetros de rotação e zoom
3. **Analise o código**: Identifique onde cada etapa do pipeline acontece
4. **Experimente arquiteturas**: Adicione ou remova camadas convolucionais

### Nível Avançado
1. **Adicione métricas**: Implemente precision, recall, F1-score
2. **Visualize features**: Extraia mapas de ativação das camadas Conv2D
3. **Matriz de confusão**: Identifique quais dígitos são mais confundidos
4. **Regularização**: Adicione Dropout ou Batch Normalization
5. **Experimente otimizadores**: Compare Adam, SGD, RMSprop

## 💡 Discussões para Sala de Aula

### Tópicos para Debate
- **Overfitting**: O que acontece se treinarmos por muitas epochs?
- **Qualidade dos Dados**: Como desenhos mal feitos afetam a predição?
- **Viés do Modelo**: O modelo favorece algum dígito específico?
- **Quantidade de Dados**: Quantos feedbacks são necessários para melhorar?
- **Ética em IA**: Implicações de sistemas que aprendem com humanos
- **Generalização**: Por que o modelo funciona com estilos diferentes de escrita?

### Experimentos Práticos
1. Desenhe propositalmente de forma ambígua (ex: 1 ou 7, 5 ou 6)
2. Teste com diferentes estilos de escrita (cursiva, impressa)
3. Compare acurácia antes e depois do re-treinamento
4. Identifique quais dígitos são mais fáceis/difíceis de classificar
5. Teste com números fora do padrão (muito grandes, muito pequenos)

## 📖 Material de Apoio

### Referências Técnicas
- **Dataset MNIST**: 70.000 imagens de dígitos manuscritos (benchmark clássico)
- **Arquitetura CNN**: Inspirada em LeNet-5 (Yann LeCun, 1998)
- **Adam Optimizer**: Adaptive Moment Estimation (eficiente e robusto)
- **Softmax**: Normaliza saídas em distribuição de probabilidade

### Perguntas Frequentes (FAQ)

**Por que 28x28 pixels?**
- Padrão do dataset MNIST, balanceia detalhes e eficiência computacional

**O que é Human-in-the-Loop?**
- Humanos fornecem feedback que o modelo usa para melhorar continuamente

**Como funciona o re-treinamento?**
- Novos exemplos do feedback são adicionados ao dataset original e o modelo é treinado novamente

**Por que usar CNN ao invés de rede neural simples?**
- CNNs detectam padrões espaciais (bordas, curvas) essenciais para visão computacional

## 🐛 Solução de Problemas

### ❌ Python 3.11 não encontrado

**Sintoma:** `python3.11: command not found`

**Solução:**
```bash
# Verifique versões disponíveis
python3 --version
python --version

# Use a versão disponível (mínimo 3.8)
python3 -m venv venv
# ou
python -m venv venv
```

### ❌ Comando 'python' não reconhecido (Windows)

**Sintoma:** `'python' is not recognized as an internal or external command`

**Solução:**
1. Reinstale o Python marcando "Add to PATH"
2. Ou adicione manualmente ao PATH:
   - Pesquise "Variáveis de Ambiente" no Windows
   - Adicione `C:\Users\SeuUsuario\AppData\Local\Programs\Python\Python311` ao PATH

### ❌ Ambiente virtual não ativa no PowerShell

**Sintoma:** `execution of scripts is disabled on this system`

**Solução:**
```powershell
# Execute como Administrador
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Tente novamente
venv\Scripts\Activate.ps1
```

### ❌ Erro ao instalar dependências

**Sintoma:** `error: Microsoft Visual C++ 14.0 or greater is required`

**Solução (Windows):**
1. Instale o [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Ou instale versões pré-compiladas:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### ❌ TensorFlow não instala no Apple Silicon (M1/M2/M3)

**Sintoma:** Erro ao instalar `tensorflow==2.15.0`

**Solução:**
```bash
# Instale versão compatível com Metal
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal
```

### ❌ Modelo não carrega

**Sintoma:** `OSError: Unable to open file`

**Solução:**
```bash
# Re-treine o modelo
python3 train_model.py
```

### ❌ Porta 5000 já está em uso

**Sintoma:** `Address already in use`

**Solução:**
```bash
# Mude a porta no app.py (última linha)
# De: app.run(debug=True)
# Para: app.run(debug=True, port=5001)

# Ou mate o processo na porta 5000
# macOS/Linux:
lsof -ti:5000 | xargs kill -9

# Windows:
netstat -ano | findstr :5000
taskkill /PID <número_do_PID> /F
```

### ❌ Predições sempre incorretas

**Sintomas:** Modelo erra constantemente

**Soluções:**
- Execute o re-treinamento: `python3 train_model.py`
- Desenhe dígitos mais claros e centralizados
- Verifique se o `model.keras` não está corrompido (delete e re-treine)
- Colete feedbacks e re-treine com novos exemplos

### ❌ VS Code não encontra o interpretador Python

**Sintoma:** "Python interpreter not found"

**Solução:**
1. Instale a extensão Python da Microsoft
2. `Cmd+Shift+P` → "Python: Select Interpreter"
3. Se não aparecer, digite o caminho manualmente:
   - macOS/Linux: `./venv/bin/python`
   - Windows: `.\venv\Scripts\python.exe`

### ❌ Import error: No module named 'tensorflow'

**Sintoma:** Módulo não encontrado mesmo após instalação

**Solução:**
```bash
# Verifique se está no ambiente virtual
which python  # macOS/Linux
where python  # Windows

# Se não mostrar o caminho da venv, ative novamente
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstale as dependências
pip install -r requirements.txt
```

### ❌ MNIST dataset não baixa

**Sintoma:** Erro ao baixar dataset durante treinamento

**Solução:**
```bash
# Verifique conexão com internet
# Ou baixe manualmente de: yann.lecun.com/exdb/mnist/
# Coloque os arquivos em: ~/.keras/datasets/
```

### 💡 Dica para Alunos

Se encontrar um erro não listado aqui:
1. Leia a mensagem de erro cuidadosamente
2. Copie a mensagem e pesquise no Google
3. Verifique o Stack Overflow
4. Pergunte ao professor ou colegas
5. Documente a solução quando resolver!

## 📈 Possíveis Melhorias Futuras

### Para Fins Educacionais
- [ ] Dashboard com métricas em tempo real (accuracy, loss por epoch)
- [ ] Visualização interativa das camadas convolucionais
- [ ] Gráficos de evolução do treinamento (loss curve, accuracy curve)
- [ ] Matriz de confusão interativa
- [ ] Comparação entre diferentes arquiteturas (LeNet, AlexNet simplificada)
- [ ] Notebook Jupyter integrado para análises
- [ ] Exportação de relatórios de performance em PDF
- [ ] Sistema de versionamento de modelos

### Funcionalidades Técnicas
- [ ] Autenticação de usuários (professores/alunos)
- [ ] Re-treinamento automático por agendamento (cron job)
- [ ] Suporte para múltiplos idiomas (i18n)
- [ ] Suporte mobile com touch events otimizado
- [ ] API REST documentada (Swagger/OpenAPI)
- [ ] Base de dados para armazenar histórico de predições
- [ ] Containerização (Docker)
- [ ] Deploy em cloud (Azure, AWS, GCP)

## 👨‍🏫 Guia para Professores

### Como Usar em Aula

**Demonstração Inicial (15-20 min)**
1. Apresente a aplicação funcionando
2. Desenhe alguns dígitos e mostre as predições
3. Demonstre o sistema de feedback
4. Execute um re-treinamento ao vivo

**Atividade Prática (30-45 min)**
1. Alunos acessam a aplicação (localmente ou em servidor)
2. Cada aluno testa com 20-30 dígitos diferentes
3. Coletam feedbacks de predições incorretas
4. Executam re-treinamento em grupo
5. Comparam performance antes/depois

**Discussão e Análise (15-20 min)**
1. Discutir resultados observados
2. Analisar padrões de erros do modelo
3. Conectar teoria com prática observada
4. Responder dúvidas dos alunos

### Sugestões de Avaliação

**Trabalho Prático**
- Modificar hiperparâmetros e documentar resultados
- Implementar melhorias na arquitetura
- Criar relatório comparativo de performance

**Projeto Final**
- Estender para reconhecer letras (A-Z)
- Adicionar visualizações de camadas
- Implementar outras arquiteturas (VGG, ResNet simplificadas)

### Pré-requisitos Recomendados

**Conhecimentos Teóricos**
- Fundamentos de Python
- Conceitos básicos de Machine Learning
- Álgebra linear (matrizes, vetores)
- Cálculo (derivadas, otimização)

**Infraestrutura**
- Python 3.8+ instalado
- Computadores com pelo menos 4GB RAM
- Acesso à internet (para instalar dependências)

## 📝 Licença

Este projeto é de código aberto e está disponível para fins educacionais no ensino superior de Ciência de Dados.

## 🤝 Contribuições

Contribuições são bem-vindas, especialmente de:
- **Professores**: Sugestões de exercícios e melhorias pedagógicas
- **Alunos**: Identificação de bugs e dúvidas comuns
- **Desenvolvedores**: Otimizações de código e novas funcionalidades
- **Pesquisadores**: Extensões para trabalhos acadêmicos

### Como Contribuir
1. Reporte bugs através de issues
2. Sugira novas funcionalidades educacionais
3. Melhore a documentação e exemplos
4. Adicione exercícios práticos
5. Otimize algoritmos de processamento
6. Crie tutoriais em vídeo/texto

## 🎓 Aplicações Acadêmicas

Esta ferramenta é ideal para:
- **Cursos de Ciência de Dados**: Módulo de Deep Learning
- **Disciplinas de Machine Learning**: Aulas práticas de CNN
- **Laboratórios de IA**: Experimentação hands-on
- **Projetos de Disciplina**: Base para trabalhos e extensões
- **TCC/Trabalhos Finais**: Ponto de partida para pesquisa aplicada
- **Workshops e Minicursos**: Demonstrações interativas

## 📞 Suporte

Para dúvidas, sugestões ou problemas:
- Abra uma issue no repositório
- Consulte a documentação completa
- Participe de discussões sobre melhorias

---

**Desenvolvido com ❤️ para o ensino de Ciência de Dados**  
**Tecnologias: Python • TensorFlow • Keras • Flask • OpenCV**

*Ferramenta educacional para demonstração prática de conceitos de Deep Learning e aprendizado contínuo*
