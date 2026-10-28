# 🚀 HACKOWEEK SEM 5 — Weekly Task Repository

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000.svg?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-blue.svg)](https://xgboost.readthedocs.io/)

> **SIT Nagpur — Hack-o-Week | 5th Semester (Third Year)**
>
> A progressive, 18-week Machine Learning and Deep Learning curriculum — from REST APIs & Python fundamentals to Ensemble Methods, PyTorch, Transformers, and LLM APIs. Each bi-weekly module contains a Jupyter Notebook, an interactive web dashboard, and a dedicated ML engine built on the **Kaggle Student Performance & Academic Insights Dataset**.

---

## 📋 Weekly Curriculum Overview

| Week | Topic | Key Skills | Deliverables |
|:----:|-------|------------|:------------:|
| **1 – 2** | [APIs & Backend Basics](WEEK1%20-%20WEEK2/) | REST API, Flask, CRUD, HTTP Methods | 🌐 📄 |
| **3 – 4** | [Python, NumPy, Pandas & Visualization](WEEK3%20-%20WEEK4/) | OOP, Arrays, DataFrames, Matplotlib, Seaborn | 📓 🌐 📄 |
| **5 – 6** | [Mathematics for ML](WEEK5%20-%20WEEK6/) | Linear Algebra, Calculus, Gradient Descent, Eigen Decomposition | 📓 🌐 📄 |
| **7 – 8** | [Supervised ML: Regression & Classification](WEEK7%20-%20WEEK8/) | Linear/Polynomial/Ridge/Lasso Regression, Logistic Regression, KNN | 📓 🌐 📄 |
| **9 – 10** | [Scikit-Learn Workflow & Clustering](WEEK9%20-%20WEEK10/) | Pipelines, Cross-Validation, ROC-AUC, K-Means, Hierarchical, DBSCAN | 📓 🌐 📄 |
| **11 – 12** | [Dimensionality Reduction](WEEK%2011%20-%20WEEK%2012/) | PCA, t-SNE, Explained Variance, Perplexity Tuning | 📓 📄 |
| **13 – 14** | [Ensemble Methods & Regularization](WEEK%2013%20-%20WEEK%2014/) | Bagging, Boosting, XGBoost, LightGBM, Bias-Variance, L1/L2 | 📓 🌐 📄 |
| **15 – 16** | [Deep Learning & Model Deployment](WEEK%2015%20-%20WEEK%2016/) | Pickle/Joblib, Perceptrons, Activations, SGD/Adam, NumPy NN | 📓 🌐 📄 |
| **17 – 18** | [Deep Learning & LLMs](WEEK%2017%20-%20WEEK%2018/) | PyTorch, Autograd, CNNs, RNNs/LSTMs, Regularization, HF, LLM APIs | 📓 🌐 📄 |

> **Legend:** 📓 Jupyter Notebook · 🌐 Interactive Web Dashboard · 📄 README Documentation

---

## 📂 Repository Structure

```
HACKOWEEK-SEM-5/
│
├── README.md                          ← You are here
│
├── WEEK1 - WEEK2/                     ← APIs & Backend Basics
│   ├── app.py                         # Flask REST API server
│   ├── test_api.py                    # API test suite
│   ├── data/                          # Kaggle Student Performance dataset (CSV + JSON)
│   └── public/                        # Interactive web dashboard
│
├── WEEK3 - WEEK4/                     ← Python, NumPy, Pandas & Visualization
│   ├── app.py                         # Flask web server
│   ├── analysis.py                    # Data analysis engine
│   ├── notebooks/                     # Jupyter notebook
│   ├── data/                          # Dataset
│   └── public/                        # Interactive dashboard
│
├── WEEK5 - WEEK6/                     ← Mathematics for ML
│   ├── app.py                         # Flask web server
│   ├── math_engine.py                 # Linear algebra & calculus engine
│   ├── notebooks/                     # Jupyter notebook
│   ├── data/                          # Dataset
│   └── public/                        # Interactive dashboard
│
├── WEEK7 - WEEK8/                     ← Supervised ML (Regression & Classification)
│   ├── app.py                         # Flask web server
│   ├── ml_engine.py                   # ML models engine
│   ├── notebooks/                     # Jupyter notebook
│   ├── data/                          # Dataset
│   └── public/                        # Interactive dashboard
│
├── WEEK9 - WEEK10/                    ← Scikit-Learn Workflow & Clustering
│   ├── app.py                         # Flask web server
│   ├── workflow_engine.py             # Pipeline & clustering engine
│   ├── notebooks/                     # Jupyter notebook
│   ├── data/                          # Dataset
│   └── public/                        # Interactive dashboard
│
├── WEEK 11 - WEEK 12/                 ← Dimensionality Reduction (PCA & t-SNE)
│   ├── dimensionality_reduction.py    # PCA & t-SNE analysis script
│   ├── notebooks/                     # Jupyter notebook
│   └── README.md
│
├── WEEK 13 - WEEK 14/                 ← Ensemble Methods & Regularization
│   ├── app.py                         # Flask web server (port 5006)
│   ├── ensemble_engine.py             # Ensemble & regularization engine
│   ├── notebooks/                     # Jupyter notebook
│   ├── data/                          # Dataset
│   └── public/                        # Interactive dashboard
│
├── WEEK 15 - WEEK 16/                 ← Deep Learning & Model Deployment
│   ├── app.py                         # Flask web server (port 5007)
│   ├── nn_deployment_engine.py        # Pure NumPy NN, optimizers, deployment engine
│   ├── notebooks/                     # Jupyter notebook
│   ├── data/                          # Dataset
│   ├── models/                        # Serialized models (.pkl, .joblib)
│   └── public/                        # Interactive dashboard
│
└── WEEK 17 - WEEK 18/                 ← Advanced Deep Learning & Generative AI
    ├── app.py                         # Flask web server (port 5008)
    ├── deep_learning_engine.py        # PyTorch engine, CNNs, RNNs, HF & LLM client
    ├── notebooks/                     # Jupyter notebook
    ├── data/                          # Dataset
    └── public/                        # Interactive dashboard
```

---

## 🧠 Topics Covered (Week-by-Week)

### Week 1 – Week 2: APIs & Backend Basics
- RESTful API design with **Flask**
- CRUD operations (Create, Read, Update, Delete)
- HTTP methods (GET, POST, PUT, DELETE) & status codes
- Student Information Management System (SIMS)
- Interactive API Explorer console
- Whiteboard Challenge: REST architecture & system design

### Week 3 – Week 4: Python, NumPy, Pandas & Visualization
- **Python Essentials**: Functions, OOP (`StudentRecord` class), comprehensions
- **NumPy**: Multidimensional arrays, broadcasting, vectorized operations, z-score normalization
- **Pandas**: DataFrames, data cleaning, `pd.merge`, multi-metric `groupby` aggregations
- **Visualization**: Matplotlib & Seaborn — KDE, boxplots, regression plots, correlation heatmaps

### Week 5 – Week 6: Mathematics for Machine Learning
- **Linear Algebra**: Vectors, matrices, dot product, norms, cosine similarity, eigenvalues & eigenvectors
- **Calculus**: Numerical & analytical derivatives, gradient vectors, **gradient descent** optimization
- **Application**: PCA via eigen decomposition on the student covariance matrix
- Chain rule for neural network backpropagation intuition

### Week 7 – Week 8: Supervised ML — Regression & Classification
- **Regression**: Linear, Polynomial (degree 2), Ridge (L₂), Lasso (L₁ & feature selection)
- **Classification**: Logistic Regression (sigmoid, decision threshold), K-Nearest Neighbors
- Metrics: MSE, RMSE, R², Accuracy, Precision, Recall, F1-Score
- Train/test split, model comparison & interpretation

### Week 9 – Week 10: Scikit-Learn Workflow & Clustering
- **Pipelines**: `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`
- **Evaluation**: Stratified K-Fold Cross-Validation, Confusion Matrix, ROC Curve, AUC
- **Clustering**: K-Means (inertia, silhouette), Hierarchical (Ward linkage), DBSCAN (density-based, noise detection)
- Elbow method & dendrogram analysis

### Week 11 – Week 12: Dimensionality Reduction
- **PCA**: Principal Component Analysis — explained variance, scree plots, 2D projection, component loadings, biplot, reconstruction error
- **t-SNE**: t-distributed Stochastic Neighbor Embedding — local neighborhood preservation, perplexity effect
- PCA vs t-SNE comparison (linear/global vs non-linear/local)
- Standardization importance & best practices

### Week 13 – Week 14: Ensemble Methods & Regularization
- **Bagging**: Bootstrap Aggregating, BaggingClassifier, **Random Forest**
- **Boosting**: Gradient Boosting, **XGBoost**, **LightGBM** (sequential error correction)
- **Bias-Variance Trade-off**: Tree depth analysis, underfitting ↔ overfitting zones
- **Overfitting & Underfitting**: Learning curves, causes, identification & prevention
- **Regularization (L1/L2)**: C-sweep analysis, coefficient comparison, L1 sparsity & feature selection

### Week 15 – Week 16: Deep Learning Fundamentals & Model Deployment
- **Model Deployment Basics**: Serializing and loading pipelines with `pickle` and `joblib`, latency & disk footprint benchmarks, security & reproducibility
- **Perceptrons & Activation Functions**: Rosenblatt single-layer perceptron, step updates, Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax, vanishing gradient dynamics
- **Loss Functions & Optimizers**: Binary Cross-Entropy (BCE) vs MSE, SGD, Momentum, Adam (adaptive moment estimation with bias correction), 2D ravine loss landscape trajectories
- **NumPy Neural Network from Scratch**: Multi-layer perceptron with He initialization, forward propagation, analytical backpropagation (chain rule), training/validation history, non-linear decision boundaries

### Week 17 – Week 18: Advanced Deep Learning & Generative AI
- **PyTorch Foundations**: Dynamic computation graphs, tensors, automatic differentiation (`autograd`), custom modular training loop
- **Convolutional Neural Networks (CNNs)**: Convolutions (`Conv2d`), spatial pooling (`MaxPool2d`), receptive fields, feature maps, spatial dimension calculus ($W_{out}$)
- **Sequence Modeling (RNN, LSTM, GRU)**: Temporal sequence processing, hidden states ($h_t$), cell memory ($c_t$), gating mechanisms (forget/input/output vs update/reset), vanishing gradient remedies
- **Regularization in Deep Learning**: Inverted Dropout ($p=0.4$), Batch Normalization (`BatchNorm1d`), Early Stopping with validation checkpoints and patience
- **Transfer Learning**: Pretrained CNN backbones (ResNet-18, MobileNet-V3), feature extraction vs fine-tuning workflows
- **Hugging Face Transformers**: Pipelines, tokenizers (WordPiece / BPE), token IDs, attention masks, text classification
- **LLM APIs & Generative AI**: OpenAI, Anthropic, Hugging Face Inference API, Ollama (local LLMs), prompt engineering for Q&A, code generation, and summarization

---

## 🗃️ Dataset

All weeks use the **Kaggle Student Performance & Academic Insights Dataset**, containing:

| Feature | Description |
|---------|-------------|
| `StudentID` | Unique identifier |
| `Name`, `Gender`, `Age` | Demographics |
| `City`, `Department`, `Year`, `Semester` | Academic info |
| `CGPA` | Cumulative GPA (0–10) |
| `AttendanceRate` | Attendance percentage |
| `StudyHoursPerWeek` | Weekly study hours |
| `MathScore`, `ReadingScore`, `WritingScore` | Subject scores |
| `Skills`, `Projects`, `Activities` | Extra-curricular data |
| `Status` | Active/Inactive |

A derived **`Distinction`** label (`CGPA ≥ 8.5`) is used as the classification target from Week 7 onwards.

---

## ⚡ Quick Start

### Prerequisites
```bash
pip install flask flask-cors scikit-learn pandas numpy matplotlib seaborn jupyter xgboost lightgbm
```

### Run any week's web dashboard
```bash
cd "WEEK<N> - WEEK<N+1>"
python app.py
# Navigate to http://localhost:500X (port varies per week)
```

### Run any week's Jupyter notebook
```bash
cd "WEEK<N> - WEEK<N+1>"
jupyter notebook notebooks/<notebook_name>.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.11 |
| **Backend** | Flask, Flask-CORS |
| **ML/Data** | scikit-learn, NumPy, Pandas, XGBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn, Chart.js (web dashboards) |
| **Notebooks** | Jupyter |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |

---

## 👩‍💻 Author

**Samruddhi Kalbande**
SIT Nagpur — 5th Semester, Third Year

---

*Built as part of the SIT-N Hack-o-Week program, 5th Semester.*
