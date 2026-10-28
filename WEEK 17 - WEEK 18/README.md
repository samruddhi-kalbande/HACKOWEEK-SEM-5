# Week 17 – Week 18: Advanced Deep Learning & Generative AI

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![LLM APIs](https://img.shields.io/badge/LLM%20APIs-OpenAI%20%7C%20Anthropic%20%7C%20Ollama-8b5cf6.svg)](https://platform.openai.com/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week17_week18_deep_learning_and_llms.ipynb)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)

Based on the official curriculum guidelines from SIT Nagpur:

1. **Intro to PyTorch (Tensors, Autograd & Training Loop)**:
   - Tensor creation, shapes, dtypes, broadcasting, and hardware acceleration (CPU vs CUDA).
   - Reverse-mode automatic differentiation (`autograd`) with dynamic computation graphs.
   - Engineering complete custom PyTorch training loops (`zero_grad()`, `forward()`, `backward()`, `step()`).
2. **Convolutional Neural Networks (CNNs)**:
   - Convolutions (`Conv2d`), spatial pooling (`MaxPool2d`), receptive fields, and feature map hierarchies.
   - Spatial dimension reduction calculus: $W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$.
   - Interactive kernel filter inspection (Sobel edge detectors, Sharpen, Gaussian blur).
3. **Sequence Modeling (RNN, LSTM, GRU)**:
   - Temporal sequence processing on student multi-term trajectory data.
   - Architectural comparison: Hidden states, memory cell $c_t$, and gating mechanisms.
   - Gating mathematics: Forget ($f_t$), Input ($i_t$), and Output ($o_t$) in LSTM vs Update ($z_t$) and Reset ($r_t$) in GRU.
   - Mitigating vanishing and exploding gradients in recurrent connections.
4. **Regularization in Deep Learning**:
   - Inverted Dropout ($p = 0.4$) preventing co-adaptation of features.
   - Batch Normalization (`BatchNorm1d` / `BatchNorm2d`) stabilizing internal covariate shift.
   - Early Stopping monitoring validation loss checkpoints to prevent catastrophic overfitting.
5. **Transfer Learning Basics (Pretrained CNNs)**:
   - Leveraging ImageNet representations via ResNet-18 and MobileNet-V3.
   - Feature Extraction (freezing backbone weights: `param.requires_grad = False`) vs. Fine-Tuning.
6. **Hugging Face Transformers Library Basics**:
   - Tokenization pipelines (WordPiece / Byte-Pair Encoding), Token IDs, and Attention Masks.
   - Pretrained NLP classification architectures (DistilBERT / RoBERTa).
7. **Using LLM APIs (OpenAI, Anthropic, Hugging Face, Ollama)**:
   - Unified multi-provider LLM API client architecture.
   - Prompt design for text generation, concept summarization, and academic Q&A.
   - Local fallback simulation engine for offline inference.

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough is available in:
[`notebooks/week17_week18_deep_learning_and_llms.ipynb`](notebooks/week17_week18_deep_learning_and_llms.ipynb)

To launch the notebook:
```bash
cd "WEEK 17 - WEEK 18"
jupyter notebook notebooks/week17_week18_deep_learning_and_llms.ipynb
```

---

## 🌐 Interactive Deep Learning & LLM Studio

An interactive, responsive dark glassmorphism dashboard has been engineered:

1. **🔥 PyTorch & Autograd**: Live interactive differentiation calculator ($y = 3x^3 + 2x^2 + 5x + 1$) and training loop simulator with dynamic BCE loss and accuracy curves.
2. **🖼️ CNNs & Convolutions**: Spatial feature map dimension walkthrough with interactive 3×3 convolution kernel visualization (Sobel, Sharpen, Gaussian Blur).
3. **🔄 Sequence Modeling**: Multi-step sequential convergence charts and trainable parameter footprint comparisons across Vanilla RNN, LSTM, and GRU.
4. **🛡️ Regularization & Transfer Learning**: Visualizing the generalization gap (Overfitting vs Dropout + BatchNorm) alongside early stopping triggers and ResNet feature extraction workflows.
5. **🤗 Transformers & LLMs**: Interactive tokenization analyzer breaking sentences into tokens/IDs, combined with a unified LLM prompt query workbench (OpenAI, Anthropic, Ollama, Local Simulation).

### How to Run the Web Dashboard:
```bash
cd "WEEK 17 - WEEK 18"
pip install flask flask-cors torch torchvision
python app.py
```
Navigate to: **`http://localhost:5008`** (or `http://127.0.0.1:5008`)

---

## 📂 Project Structure
```
WEEK 17 - WEEK 18/
├── app.py                          # Flask web server (port 5008)
├── deep_learning_engine.py         # PyTorch engine, CNNs, RNNs/LSTMs, Transformers & LLM client
├── README.md                       # Comprehensive module documentation
├── data/
│   └── kaggle_student_performance.csv
├── notebooks/
│   └── week17_week18_deep_learning_and_llms.ipynb
└── public/
    ├── index.html                  # Interactive 5-tab dashboard
    ├── css/
    │   └── styles.css              # Dark glassmorphism theme
    └── js/
        └── app.js                  # Client-side Chart.js rendering & API calls
```

---

## 🧮 Mathematical Formulations

### 1. Reverse-Mode Autograd (Vector-Jacobian Product)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{J}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \left( \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

### 2. Spatial Dimension of Convolutional Layers
$$W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$$

### 3. LSTM Gating Formulations
$$\mathbf{f}_t = \sigma(W_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget Gate)}$$
$$\mathbf{i}_t = \sigma(W_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input Gate)}$$
$$\tilde{\mathbf{c}}_t = \tanh(W_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate Cell State)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Updated Cell State)}$$
$$\mathbf{o}_t = \sigma(W_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output Gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden State)}$$

### 4. GRU Gating Formulations
$$\mathbf{z}_t = \sigma(W_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Update Gate)}$$
$$\mathbf{r}_t = \sigma(W_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Reset Gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(W \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Candidate Hidden State)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(Final Hidden State)}$$

### 5. Inverted Dropout
$$\mathbf{y} = \frac{1}{1 - p} (\mathbf{x} \odot \mathbf{m}), \quad \mathbf{m} \sim \text{Bernoulli}(1 - p)$$

### 6. Batch Normalization
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad \mathbf{y} = \gamma \hat{\mathbf{x}} + \beta$$

---

## 👩‍💻 Author
**Samruddhi Kalbande**  
SIT Nagpur — 5th Semester, Third Year  
*Hack-o-Week Program*
