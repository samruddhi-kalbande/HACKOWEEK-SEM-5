import os
import time
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    'Age', 'StudyHoursPerWeek', 'AttendanceRate',
    'MathScore', 'ReadingScore', 'WritingScore'
]
CATEGORICAL_FEATURES = ['Gender', 'Department']


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset & Preprocessing Helper
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    if not os.path.exists(DATA_FILE):
        # Fallback to week 13-14 data if relative path is missing
        alt_path = os.path.join(BASE_DIR, '..', 'WEEK 13 - WEEK 14', 'data', 'kaggle_student_performance.csv')
        df = pd.read_csv(alt_path)
    else:
        df = pd.read_csv(DATA_FILE)
    df['Distinction'] = (df['CGPA'] >= 8.5).astype(int)
    return df


def get_preprocessed_data():
    df = load_dataset()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Distinction'].values

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), NUMERIC_FEATURES),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ]
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    return {
        'preprocessor': preprocessor,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': NUMERIC_FEATURES + [f"Cat_{i}" for i in range(X_train_proc.shape[1] - len(NUMERIC_FEATURES))]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Deployment Basics (Pickle vs Joblib)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_deployment_serialization():
    data = get_preprocessed_data()
    X_train = data['X_train_raw']
    y_train = data['y_train']
    X_test = data['X_test_raw']
    y_test = data['y_test']

    # Train an end-to-end pipeline model
    pipeline = Pipeline([
        ('preprocessor', data['preprocessor']),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    pickle_path = os.path.join(MODELS_DIR, 'student_model.pkl')
    joblib_path = os.path.join(MODELS_DIR, 'student_model.joblib')

    # Benchmark Pickle Save
    t0 = time.perf_counter()
    with open(pickle_path, 'wb') as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_save_time_ms = round((time.perf_counter() - t0) * 1000, 3)

    # Benchmark Joblib Save
    t0 = time.perf_counter()
    joblib.dump(pipeline, joblib_path, compress=3)
    joblib_save_time_ms = round((time.perf_counter() - t0) * 1000, 3)

    # Benchmark Pickle Load
    t0 = time.perf_counter()
    with open(pickle_path, 'rb') as f:
        loaded_pickle_model = pickle.load(f)
    pickle_load_time_ms = round((time.perf_counter() - t0) * 1000, 3)

    # Benchmark Joblib Load
    t0 = time.perf_counter()
    loaded_joblib_model = joblib.load(joblib_path)
    joblib_load_time_ms = round((time.perf_counter() - t0) * 1000, 3)

    pickle_size_bytes = os.path.getsize(pickle_path)
    joblib_size_bytes = os.path.getsize(joblib_path)

    # Evaluate predictions
    y_pred_orig = pipeline.predict(X_test)
    y_pred_pickle = loaded_pickle_model.predict(X_test)
    y_pred_joblib = loaded_joblib_model.predict(X_test)

    pickle_valid = bool(np.array_equal(y_pred_orig, y_pred_pickle))
    joblib_valid = bool(np.array_equal(y_pred_orig, y_pred_joblib))

    test_acc = round(float(accuracy_score(y_test, y_pred_orig) * 100), 2)
    test_f1 = round(float(f1_score(y_test, y_pred_orig, zero_division=0) * 100), 2)

    return {
        "pickle": {
            "save_time_ms": pickle_save_time_ms,
            "load_time_ms": pickle_load_time_ms,
            "file_size_kb": round(pickle_size_bytes / 1024, 2),
            "valid_predictions": pickle_valid,
            "format": "Standard Python Binary (.pkl)",
            "protocol": f"Protocol {pickle.HIGHEST_PROTOCOL}",
            "compression": "None (raw binary)"
        },
        "joblib": {
            "save_time_ms": joblib_save_time_ms,
            "load_time_ms": joblib_load_time_ms,
            "file_size_kb": round(joblib_size_bytes / 1024, 2),
            "valid_predictions": joblib_valid,
            "format": "Optimized NumPy/SciPy Binary (.joblib)",
            "protocol": "Joblib Engine with zlib compression (level 3)",
            "compression": "zlib level 3"
        },
        "model_performance": {
            "accuracy": test_acc,
            "f1_score": test_f1,
            "num_test_samples": len(y_test)
        },
        "comparison_insights": [
            "Joblib uses disk caching and memory mapping, making it far superior for models storing large NumPy weight matrices.",
            "Pickle is built into Python standard library, offering universal zero-dependency compatibility across standard environments.",
            "Joblib provides transparent compression (zlib, gzip, bz2, lzma) reducing serialized model footprint on disk.",
            "Both serializers execute arbitrary code when unpickling; untrusted models should never be loaded without signature verification in production."
        ]
    }


def deploy_predict(features_dict, serializer='joblib'):
    """Load serialized model via chosen serializer and run single-sample inference."""
    model_path = os.path.join(MODELS_DIR, f'student_model.{serializer}')
    if not os.path.exists(model_path):
        benchmark_deployment_serialization()

    t0 = time.perf_counter()
    if serializer == 'pickle':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = joblib.load(model_path)
    load_latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    input_df = pd.DataFrame([features_dict])
    t_inf = time.perf_counter()
    pred = int(model.predict(input_df)[0])
    prob = model.predict_proba(input_df)[0]
    inference_latency_ms = round((time.perf_counter() - t_inf) * 1000, 3)

    return {
        "prediction": pred,
        "prediction_label": "Distinction (CGPA >= 8.5)" if pred == 1 else "Standard (CGPA < 8.5)",
        "probability_distinction": round(float(prob[1]) * 100, 2),
        "probability_standard": round(float(prob[0]) * 100, 2),
        "load_latency_ms": load_latency_ms,
        "inference_latency_ms": inference_latency_ms,
        "serializer_used": serializer
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Perceptron, Activation Functions & Derivatives
# ─────────────────────────────────────────────────────────────────────────────

class PerceptronScratch:
    """Single Layer Perceptron from scratch with Frank Rosenblatt's step update."""
    def __init__(self, learning_rate=0.01, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0.0 else 0
                update = self.lr * (target - y_pred)
                if update != 0.0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            self.history.append({"epoch": epoch, "misclassifications": errors})
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)


def get_activation_curves():
    """Generates activation functions and their analytical derivatives."""
    z = np.linspace(-5.0, 5.0, 100)

    # 1. Sigmoid
    sig = 1.0 / (1.0 + np.exp(-z))
    sig_prime = sig * (1.0 - sig)

    # 2. ReLU
    relu = np.maximum(0.0, z)
    relu_prime = np.where(z > 0.0, 1.0, 0.0)

    # 3. Leaky ReLU (alpha = 0.1)
    leaky_relu = np.where(z > 0.0, z, 0.1 * z)
    leaky_relu_prime = np.where(z > 0.0, 1.0, 0.1)

    # 4. Tanh
    tanh = np.tanh(z)
    tanh_prime = 1.0 - tanh**2

    # 5. Softmax on simulated 3-class logits
    z_multi = np.array([-2.0, 0.0, 2.0])
    exp_z = np.exp(z_multi - np.max(z_multi))
    softmax_demo = (exp_z / np.sum(exp_z)).tolist()

    return {
        "z_values": np.round(z, 2).tolist(),
        "functions": {
            "sigmoid": {
                "name": "Sigmoid (Logistic)",
                "formula": "σ(z) = 1 / (1 + e^(-z))",
                "derivative_formula": "σ'(z) = σ(z) · (1 - σ(z))",
                "range": "(0, 1)",
                "output": np.round(sig, 4).tolist(),
                "derivative": np.round(sig_prime, 4).tolist(),
                "pros": "Smooth gradient, maps inputs into probability range [0, 1].",
                "cons": "Suffers from vanishing gradient problem for |z| > 3; output not zero-centered."
            },
            "relu": {
                "name": "Rectified Linear Unit (ReLU)",
                "formula": "f(z) = max(0, z)",
                "derivative_formula": "f'(z) = 1 if z > 0 else 0",
                "range": "[0, ∞)",
                "output": np.round(relu, 4).tolist(),
                "derivative": np.round(relu_prime, 4).tolist(),
                "pros": "Computationally efficient, no vanishing gradient for positive inputs, sparse activation.",
                "cons": "'Dying ReLU' problem when activations become permanently zero for negative inputs."
            },
            "leaky_relu": {
                "name": "Leaky ReLU (α = 0.1)",
                "formula": "f(z) = z if z > 0 else αz",
                "derivative_formula": "f'(z) = 1 if z > 0 else α",
                "range": "(-∞, ∞)",
                "output": np.round(leaky_relu, 4).tolist(),
                "derivative": np.round(leaky_relu_prime, 4).tolist(),
                "pros": "Fixes the 'Dying ReLU' issue by ensuring small non-zero gradient (α) for negative inputs.",
                "cons": "Hyperparameter α must be chosen or tuned empirically."
            },
            "tanh": {
                "name": "Hyperbolic Tangent (Tanh)",
                "formula": "tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))",
                "derivative_formula": "tanh'(z) = 1 - tanh²(z)",
                "range": "(-1, 1)",
                "output": np.round(tanh, 4).tolist(),
                "derivative": np.round(tanh_prime, 4).tolist(),
                "pros": "Zero-centered output (-1 to 1), stronger gradients than sigmoid near zero.",
                "cons": "Still suffers from vanishing gradients at extreme values."
            }
        },
        "softmax_demo": {
            "logits": [-2.0, 0.0, 2.0],
            "probabilities": [round(p, 4) for p in softmax_demo],
            "formula": "Softmax(z_i) = exp(z_i) / Σ exp(z_j)",
            "use_case": "Final layer for multi-class classification, ensuring all class outputs sum to 1.0."
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss Functions & Optimizers (SGD, Momentum, Adam)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_optimizers():
    """Simulates 2D loss surface convergence trajectories for SGD, Momentum, and Adam."""
    # Loss surface: anisotropic bowl f(x, y) = 0.5 * (x^2 + 10 * y^2) (ravine-like)
    def loss(x, y):
        return 0.5 * (x**2 + 10.0 * (y**2))

    def grad(x, y):
        return np.array([x, 10.0 * y])

    start_point = np.array([-4.0, 3.0])
    n_steps = 40

    # 1. Vanilla SGD
    lr_sgd = 0.15
    path_sgd = [start_point.copy()]
    curr_sgd = start_point.copy()
    for _ in range(n_steps):
        g = grad(curr_sgd[0], curr_sgd[1])
        curr_sgd -= lr_sgd * g
        path_sgd.append(curr_sgd.copy())

    # 2. SGD with Momentum (beta = 0.9)
    lr_mom = 0.15
    beta = 0.9
    v = np.zeros(2)
    path_mom = [start_point.copy()]
    curr_mom = start_point.copy()
    for _ in range(n_steps):
        g = grad(curr_mom[0], curr_mom[1])
        v = beta * v + (1.0 - beta) * g
        curr_mom -= lr_mom * v
        path_mom.append(curr_mom.copy())

    # 3. Adam Optimizer
    lr_adam = 0.35
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    m = np.zeros(2)
    v_adam = np.zeros(2)
    path_adam = [start_point.copy()]
    curr_adam = start_point.copy()
    for t in range(1, n_steps + 1):
        g = grad(curr_adam[0], curr_adam[1])
        m = beta1 * m + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g ** 2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        curr_adam -= (lr_adam / (np.sqrt(v_hat) + eps)) * m_hat
        path_adam.append(curr_adam.copy())

    # Format loss histories
    history_sgd = [round(float(loss(p[0], p[1])), 4) for p in path_sgd]
    history_mom = [round(float(loss(p[0], p[1])), 4) for p in path_mom]
    history_adam = [round(float(loss(p[0], p[1])), 4) for p in path_adam]

    return {
        "steps": list(range(n_steps + 1)),
        "loss_histories": {
            "SGD": history_sgd,
            "Momentum": history_mom,
            "Adam": history_adam
        },
        "final_losses": {
            "SGD": history_sgd[-1],
            "Momentum": history_mom[-1],
            "Adam": history_adam[-1]
        },
        "trajectories_2d": {
            "SGD": [{"x": round(float(p[0]), 3), "y": round(float(p[1]), 3)} for p in path_sgd],
            "Momentum": [{"x": round(float(p[0]), 3), "y": round(float(p[1]), 3)} for p in path_mom],
            "Adam": [{"x": round(float(p[0]), 3), "y": round(float(p[1]), 3)} for p in path_adam]
        },
        "explanations": {
            "SGD": "Follows instantaneous gradient direction. Oscillates violently across steep ravines (y-axis) while making slow progress along flatter dimensions.",
            "Momentum": "Accumulates past velocity vector (exponentially weighted average), dampening perpendicular oscillations and accelerating along consistent gradients.",
            "Adam": "Adaptive Moment Estimation combines Momentum (1st raw moment) with RMSProp (2nd uncentered moment) and bias correction, automatically scaling step sizes per parameter."
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pure NumPy Neural Network from Scratch
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNetworkScratch:
    """
    Modular Multi-Layer Perceptron implemented purely in NumPy.
    Supports forward propagation, backpropagation, BCE loss, SGD & Adam optimizers,
    and multiple activation functions.
    """
    def __init__(self, layer_dims, activation='relu', optimizer='adam', lr=0.01):
        self.layer_dims = layer_dims  # e.g. [input_dim, 16, 8, 1]
        self.activation_name = activation.lower()
        self.optimizer_name = optimizer.lower()
        self.lr = lr
        self.parameters = {}
        self.v = {}  # Momentum / 1st moment
        self.s = {}  # RMSProp / 2nd moment
        self.t = 0   # Adam timestep
        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(42)
        L = len(self.layer_dims)
        for l in range(1, L):
            n_in = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]
            # He initialization for ReLU / Xavier for Tanh & Sigmoid
            if self.activation_name in ['relu', 'leaky_relu']:
                factor = np.sqrt(2.0 / n_in)
            else:
                factor = np.sqrt(1.0 / n_in)

            self.parameters[f'W{l}'] = np.random.randn(n_out, n_in) * factor
            self.parameters[f'b{l}'] = np.zeros((n_out, 1))

            # Optimizer state
            self.v[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.v[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            self.s[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.s[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])

    def _apply_activation(self, Z, act_name):
        if act_name == 'relu':
            return np.maximum(0.0, Z)
        elif act_name == 'leaky_relu':
            return np.where(Z > 0.0, Z, 0.01 * Z)
        elif act_name == 'tanh':
            return np.tanh(Z)
        elif act_name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -25.0, 25.0)))
        return Z

    def _activation_derivative(self, Z, act_name):
        if act_name == 'relu':
            return np.where(Z > 0.0, 1.0, 0.0)
        elif act_name == 'leaky_relu':
            return np.where(Z > 0.0, 1.0, 0.01)
        elif act_name == 'tanh':
            return 1.0 - np.tanh(Z)**2
        elif act_name == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-np.clip(Z, -25.0, 25.0)))
            return s * (1.0 - s)
        return np.ones_like(Z)

    def forward(self, X):
        """
        Forward propagation across L layers.
        X shape: (n_features, n_samples)
        Returns output A_L and cache of (A_prev, W, b, Z)
        """
        caches = []
        A = X
        L = len(self.layer_dims) - 1

        # Hidden layers
        for l in range(1, L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            A = self._apply_activation(Z, self.activation_name)
            caches.append((A_prev, W, b, Z, self.activation_name))

        # Output layer (Binary Classification: Sigmoid)
        A_prev = A
        W_L = self.parameters[f'W{L}']
        b_L = self.parameters[f'b{L}']
        Z_L = np.dot(W_L, A_prev) + b_L
        A_L = self._apply_activation(Z_L, 'sigmoid')
        caches.append((A_prev, W_L, b_L, Z_L, 'sigmoid'))

        return A_L, caches

    def compute_loss(self, A_L, Y):
        """Binary Cross-Entropy Loss."""
        m = Y.shape[1]
        eps = 1e-12
        A_clipped = np.clip(A_L, eps, 1.0 - eps)
        cost = - (1.0 / m) * np.sum(Y * np.log(A_clipped) + (1.0 - Y) * np.log(1.0 - A_clipped))
        return float(np.squeeze(cost))

    def backward(self, A_L, Y, caches):
        """
        Backward propagation through chain rule.
        Returns parameter gradients dW, db.
        """
        grads = {}
        L = len(caches)
        m = Y.shape[1]
        Y = Y.reshape(A_L.shape)

        # Output layer gradient: dZ_L = A_L - Y (for BCE + Sigmoid)
        dZ = A_L - Y
        A_prev, W, b, Z, act = caches[L - 1]
        grads[f'dW{L}'] = (1.0 / m) * np.dot(dZ, A_prev.T)
        grads[f'db{L}'] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        # Propagate backwards through hidden layers
        for l in reversed(range(1, L)):
            A_prev, W, b, Z, act = caches[l - 1]
            dZ = dA_prev * self._activation_derivative(Z, act)
            grads[f'dW{l}'] = (1.0 / m) * np.dot(dZ, A_prev.T)
            grads[f'db{l}'] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads):
        L = len(self.layer_dims) - 1
        self.t += 1

        if self.optimizer_name == 'sgd':
            for l in range(1, L + 1):
                self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
                self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']

        elif self.optimizer_name == 'momentum':
            beta = 0.9
            for l in range(1, L + 1):
                self.v[f'dW{l}'] = beta * self.v[f'dW{l}'] + (1.0 - beta) * grads[f'dW{l}']
                self.v[f'db{l}'] = beta * self.v[f'db{l}'] + (1.0 - beta) * grads[f'db{l}']
                self.parameters[f'W{l}'] -= self.lr * self.v[f'dW{l}']
                self.parameters[f'b{l}'] -= self.lr * self.v[f'db{l}']

        elif self.optimizer_name == 'adam':
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            for l in range(1, L + 1):
                # 1st moment
                self.v[f'dW{l}'] = beta1 * self.v[f'dW{l}'] + (1.0 - beta1) * grads[f'dW{l}']
                self.v[f'db{l}'] = beta1 * self.v[f'db{l}'] + (1.0 - beta1) * grads[f'db{l}']
                # 2nd moment
                self.s[f'dW{l}'] = beta2 * self.s[f'dW{l}'] + (1.0 - beta2) * (grads[f'dW{l}'] ** 2)
                self.s[f'db{l}'] = beta2 * self.s[f'db{l}'] + (1.0 - beta2) * (grads[f'db{l}'] ** 2)
                # Bias corrected
                v_corrected_W = self.v[f'dW{l}'] / (1.0 - beta1 ** self.t)
                v_corrected_b = self.v[f'db{l}'] / (1.0 - beta1 ** self.t)
                s_corrected_W = self.s[f'dW{l}'] / (1.0 - beta2 ** self.t)
                s_corrected_b = self.s[f'db{l}'] / (1.0 - beta2 ** self.t)

                self.parameters[f'W{l}'] -= (self.lr / (np.sqrt(s_corrected_W) + eps)) * v_corrected_W
                self.parameters[f'b{l}'] -= (self.lr / (np.sqrt(s_corrected_b) + eps)) * v_corrected_b

    def fit(self, X_train, y_train, epochs=100, X_val=None, y_val=None):
        """
        Train neural net.
        X_train shape: (n_features, n_samples)
        y_train shape: (1, n_samples)
        """
        history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(1, epochs + 1):
            # Forward
            A_L, caches = self.forward(X_train)
            loss_t = self.compute_loss(A_L, y_train)

            # Backward
            grads = self.backward(A_L, y_train, caches)

            # Update
            self.update_parameters(grads)

            # Evaluate training accuracy
            y_pred_train = (A_L >= 0.5).astype(int)
            acc_t = float(np.mean(y_pred_train == y_train)) * 100.0

            val_cost = None
            acc_v = None
            if X_val is not None and y_val is not None:
                A_val, _ = self.forward(X_val)
                val_cost = self.compute_loss(A_val, y_val)
                y_pred_val = (A_val >= 0.5).astype(int)
                acc_v = float(np.mean(y_pred_val == y_val)) * 100.0

            if epoch % max(1, epochs // 30) == 0 or epoch == epochs or epoch == 1:
                history["epochs"].append(epoch)
                history["train_loss"].append(round(loss_t, 4))
                history["train_acc"].append(round(acc_t, 2))
                if val_cost is not None:
                    history["val_loss"].append(round(val_cost, 4))
                    history["val_acc"].append(round(acc_v, 2))

        return history

    def predict_proba(self, X):
        A_L, _ = self.forward(X)
        return A_L

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training Runner & High-Level API Methods
# ─────────────────────────────────────────────────────────────────────────────

def train_neural_network_experiment(epochs=120, lr=0.03, activation='relu', optimizer='adam', hidden_units=16):
    """Trains NumPy Neural Network on the real Kaggle Student Performance Dataset."""
    data = get_preprocessed_data()
    X_train = data['X_train_proc'].T  # Shape: (n_features, n_samples)
    y_train = data['y_train'].reshape(1, -1)
    X_test = data['X_test_proc'].T
    y_test = data['y_test'].reshape(1, -1)

    n_features = X_train.shape[0]
    layer_dims = [n_features, int(hidden_units), int(hidden_units // 2) if hidden_units >= 8 else 4, 1]

    t0 = time.perf_counter()
    nn = NeuralNetworkScratch(layer_dims=layer_dims, activation=activation, optimizer=optimizer, lr=float(lr))
    history = nn.fit(X_train, y_train, epochs=int(epochs), X_val=X_test, y_val=y_test)
    train_duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Test evaluation
    test_probs = nn.predict_proba(X_test)[0]
    test_preds = (test_probs >= 0.5).astype(int)
    y_test_flat = y_test[0]

    acc = round(float(accuracy_score(y_test_flat, test_preds) * 100), 2)
    prec = round(float(precision_score(y_test_flat, test_preds, zero_division=0) * 100), 2)
    rec = round(float(recall_score(y_test_flat, test_preds, zero_division=0) * 100), 2)
    f1 = round(float(f1_score(y_test_flat, test_preds, zero_division=0) * 100), 2)

    try:
        auc = round(float(roc_auc_score(y_test_flat, test_probs) * 100), 2)
    except Exception:
        auc = 50.0

    cm = confusion_matrix(y_test_flat, test_preds).tolist()

    return {
        "architecture": {
            "layers": layer_dims,
            "activation": activation.upper(),
            "optimizer": optimizer.upper(),
            "learning_rate": lr,
            "epochs": epochs
        },
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "train_duration_ms": train_duration_ms
        },
        "history": history
    }


def compute_decision_boundary():
    """Generates synthetic 2D non-linear classification dataset and evaluates NumPy NN decision surface."""
    np.random.seed(42)
    # Generate non-linear concentric circles / moons
    n_samples = 300
    r1 = np.random.uniform(0.0, 1.8, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)

    r2 = np.random.uniform(2.4, 4.0, n_samples // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)

    X = np.vstack([X1, X2])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Standardize
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    # Train NumPy NN
    nn = NeuralNetworkScratch(layer_dims=[2, 12, 6, 1], activation='relu', optimizer='adam', lr=0.05)
    nn.fit(X_norm.T, y.reshape(1, -1), epochs=200)

    # Grid for contour
    grid_res = 35
    x_span = np.linspace(X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5, grid_res)
    y_span = np.linspace(X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5, grid_res)
    xx, yy = np.meshgrid(x_span, y_span)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    probs = nn.predict_proba(grid_points.T)[0]
    preds = (probs >= 0.5).astype(int)

    # Sample scatter points
    sample_points = [
        {"x": round(float(X_norm[i, 0]), 3), "y": round(float(X_norm[i, 1]), 3), "label": int(y[i])}
        for i in range(len(y))
    ]

    return {
        "scatter_points": sample_points,
        "grid_resolution": grid_res,
        "grid": [
            {"x": round(float(grid_points[i, 0]), 2), "y": round(float(grid_points[i, 1]), 2), "prob": round(float(probs[i]), 3), "pred": int(preds[i])}
            for i in range(len(probs))
        ]
    }


if __name__ == '__main__':
    print("[1] Benchmarking Deployment Serialization (Pickle vs Joblib)...")
    res_dep = benchmark_deployment_serialization()
    print("Pickle vs Joblib:", res_dep['pickle']['file_size_kb'], "KB vs", res_dep['joblib']['file_size_kb'], "KB")

    print("\n[2] Testing Pure NumPy Neural Net from Scratch...")
    res_nn = train_neural_network_experiment(epochs=50, lr=0.03, activation='relu', optimizer='adam')
    print("Scratch NN Test Accuracy:", res_nn['metrics']['accuracy'], "% | F1:", res_nn['metrics']['f1_score'], "%")

    print("\n[3] Testing Optimizers Simulation...")
    res_opt = simulate_optimizers()
    print("Final Losses - SGD:", res_opt['final_losses']['SGD'], "| Momentum:", res_opt['final_losses']['Momentum'], "| Adam:", res_opt['final_losses']['Adam'])
