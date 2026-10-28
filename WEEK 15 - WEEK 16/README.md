# Week 15 – Week 16: Deep Learning Fundamentals & Model Deployment

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-MLP%20from%20Scratch-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Deployment](https://img.shields.io/badge/Model%20Deployment-Pickle%20%7C%20Joblib-8b5cf6.svg)](https://joblib.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week15_week16_neural_networks_and_deployment.ipynb)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)

Based on the official curriculum guidelines from SIT Nagpur:

1. **Model Deployment Basics (`pickle` & `joblib`)**:
   - Save and load trained machine learning pipelines using `pickle` and `joblib`.
   - In-depth benchmarking of serialization latency, deserialization cold-start time, and file size on disk.
   - End-to-end pipeline packaging (data preprocessors + feature scalers + trained estimators) to avoid train-serving skew.
   - Production deployment security considerations (unpickling untrusted byte streams, environment pinning).
2. **Perceptrons, Activation Functions & Propagation**:
   - Frank Rosenblatt's Single-Layer Perceptron architecture and error-driven weight update rule.
   - Non-linear activation functions: Sigmoid ($\sigma$), ReLU, Leaky ReLU, Tanh, and Softmax.
   - Vanishing gradient analysis and dying ReLU mitigation through analytical derivative comparisons.
   - Forward propagation matrix formulations and backward propagation gradient derivation via the chain rule.
3. **Loss Functions & Optimization Algorithms**:
   - Binary Cross-Entropy (BCE) Loss derivation: $L = -\frac{1}{m} \sum [y \ln \hat{y} + (1-y) \ln(1-\hat{y})]$.
   - Mean Squared Error (MSE) comparison and gradient behaviors.
   - Comparative analysis of optimizers:
     - Vanilla Stochastic Gradient Descent (SGD)
     - SGD with Momentum ($\beta = 0.9$)
     - Adam (Adaptive Moment Estimation with bias-corrected 1st & 2nd moments).
   - 2D loss surface contour simulation along high-curvature ravines.
4. **Building a Neural Network from Scratch in Pure NumPy**:
   - Fully vectorized modular Multi-Layer Perceptron implemented purely in NumPy (zero deep learning frameworks).
   - He Normal and Xavier weight initializations.
   - Forward pass across arbitrary hidden layer depths with intermediate cache storage.
   - Analytical backward pass computing parameter gradients ($dZ, dW, db$).
   - Training on the Kaggle Student Performance Dataset to classify academic `Distinction` ($CGPA \ge 8.5$).
   - Non-linear decision boundary boundary separation on synthetic concentric datasets.

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough is available in:
[`notebooks/week15_week16_neural_networks_and_deployment.ipynb`](notebooks/week15_week16_neural_networks_and_deployment.ipynb)

To launch the notebook:
```bash
cd "WEEK 15 - WEEK 16"
jupyter notebook notebooks/week15_week16_neural_networks_and_deployment.ipynb
```

---

## 🌐 Interactive Neural Networks & Deployment Studio

An interactive, responsive dark glassmorphism dashboard has been engineered:

1. **📦 Model Deployment**: Live benchmarking of Pickle vs Joblib with interactive metric cards, latency comparison bar charts, and production deployment best practices.
2. **⚡ NumPy Neural Net Simulator**: Interactive training workbench allowing real-time adjustment of Epochs, Learning Rate, Hidden Neurons, Activation functions, and Optimizers with live loss/accuracy progression curves and confusion matrix readouts.
3. **📈 Activations & Perceptron**: Interactive curve visualizer comparing functions and their analytical derivatives, with vanishing gradient insights and single-layer perceptron logic.
4. **🎯 Loss & Optimizers**: Simulated 2D ravine optimization path visualizer tracking convergence trajectories of SGD vs Momentum vs Adam.
5. **🔮 Live Inference**: Real-time student performance classification serving predictions loaded on-the-fly via serialized Joblib / Pickle models.

### How to Run the Web Dashboard:
```bash
cd "WEEK 15 - WEEK 16"
pip install flask flask-cors scikit-learn pandas numpy joblib
python app.py
```
Navigate to: **`http://localhost:5007`** (or `http://127.0.0.1:5007`)

---

## 📂 Project Structure
```
WEEK 15 - WEEK 16/
├── app.py                          # Flask web server (port 5007)
├── nn_deployment_engine.py         # Pure NumPy NN, optimizers, serialization benchmarks
├── README.md                       # Comprehensive module documentation
├── data/
│   └── kaggle_student_performance.csv
├── models/                         # Serialized pipeline artifacts (.pkl & .joblib)
├── notebooks/
│   └── week15_week16_neural_networks_and_deployment.ipynb
└── public/
    ├── index.html                  # Interactive dashboard
    ├── css/
    │   └── styles.css              # Dark glassmorphism theme
    └── js/
        └── app.js                  # Client-side Chart.js rendering & API calls
```

---

## 🧮 Mathematical Formulations

### 1. He Normal Initialization
$$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n^{[l-1]}}}\right), \quad b^{[l]} = \mathbf{0}$$

### 2. Forward Propagation
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

### 3. Binary Cross-Entropy Loss
$$\mathcal{L}(A^{[L]}, Y) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(a^{[L](i)}) + (1 - y^{(i)}) \ln(1 - a^{[L](i)}) \right]$$

### 4. Backward Propagation (Chain Rule)
$$dZ^{[L]} = A^{[L]} - Y$$
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$
$$db^{[l]} = \frac{1}{m} \sum_{i=1}^m dZ^{[l](i)}$$
$$dZ^{[l-1]} = \left( (W^{[l]})^T dZ^{[l]} \right) \odot {g'}^{[l-1]}(Z^{[l-1]})$$

### 5. Adam Optimizer Parameter Update
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 👩‍💻 Author
**Samruddhi Kalbande**  
SIT Nagpur — 5th Semester, Third Year  
*Hack-o-Week Program*
