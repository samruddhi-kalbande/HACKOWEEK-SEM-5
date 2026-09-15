import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')

# ─── 1. Vector Operations & Dot Product ─────────────────────────────────────

def vector_operations(u, v):
    """
    Computes vector norms, dot product, and cosine similarity.
    u, v: lists or 1D arrays of equal length.
    """
    u_arr = np.array(u, dtype=float)
    v_arr = np.array(v, dtype=float)

    norm_u = float(np.linalg.norm(u_arr))
    norm_v = float(np.linalg.norm(v_arr))
    dot_prod = float(np.dot(u_arr, v_arr))

    if norm_u == 0 or norm_v == 0:
        cos_sim = 0.0
        angle_deg = 90.0
    else:
        cos_sim = float(np.clip(dot_prod / (norm_u * norm_v), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))

    return {
        "u": u_arr.tolist(),
        "v": v_arr.tolist(),
        "norm_u": round(norm_u, 4),
        "norm_v": round(norm_v, 4),
        "dot_product": round(dot_prod, 4),
        "cosine_similarity": round(cos_sim, 4),
        "angle_degrees": round(angle_deg, 2)
    }

# ─── 2. 2D Matrix Transformations ──────────────────────────────────────────

def apply_2d_transformation(matrix_2x2, points=None):
    """
    Applies a 2x2 linear transformation matrix to sample coordinate points.
    """
    M = np.array(matrix_2x2, dtype=float)
    if points is None:
        # Unit square vertices + diagonal vectors
        points = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 1]]

    pts = np.array(points, dtype=float).T  # Shape (2, N)
    transformed = (M @ pts).T  # Shape (N, 2)

    det_M = float(np.linalg.det(M))

    return {
        "matrix": M.tolist(),
        "determinant": round(det_M, 4),
        "area_scaling_factor": abs(round(det_M, 4)),
        "is_invertible": abs(det_M) > 1e-6,
        "original_points": points,
        "transformed_points": np.round(transformed, 3).tolist()
    }

# ─── 3. Covariance Matrix, Eigenvalues & Eigenvectors ────────────────────────

def compute_eigen_analysis():
    """
    Computes the covariance matrix, eigenvalues, eigenvectors, and explained variance
    for student quantitative features from the Kaggle dataset.
    """
    df = pd.read_csv(DATA_FILE)
    features = ['MathScore', 'ReadingScore', 'WritingScore']
    X = df[features].to_numpy(dtype=float)

    # 1. Mean-centering
    X_centered = X - np.mean(X, axis=0)

    # 2. Covariance matrix: Cov(X) = (1 / (N - 1)) * (X^T * X)
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 3. Eigen decomposition: Sigma * v = lambda * v
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var

    return {
        "features": features,
        "covariance_matrix": np.round(cov_matrix, 2).tolist(),
        "eigenvalues": [round(float(val), 3) for val in eigenvalues],
        "explained_variance_ratio": [round(float(r) * 100, 1) for r in explained_variance_ratio],
        "eigenvectors": [np.round(eigenvectors[:, i], 3).tolist() for i in range(len(features))],
        "pc1_dominance_percentage": round(float(explained_variance_ratio[0]) * 100, 1)
    }

# ─── 4. Gradient Descent Optimization for Linear Regression ──────────────────

def run_gradient_descent(learning_rate=0.01, epochs=100):
    """
    Fits Linear Regression: CGPA = w * StudyHours + b using Gradient Descent.
    Operates on the Kaggle dataset.
    """
    df = pd.read_csv(DATA_FILE)
    X = df['StudyHoursPerWeek'].to_numpy(dtype=float)
    y = df['CGPA'].to_numpy(dtype=float)

    m = len(X)
    # Feature scaling for stable convergence
    x_mean, x_std = np.mean(X), np.std(X)
    X_scaled = (X - x_mean) / x_std

    w = 0.0
    b = float(np.mean(y))

    history = []
    for epoch in range(1, epochs + 1):
        y_pred = w * X_scaled + b
        loss = float((1 / (2 * m)) * np.sum((y_pred - y) ** 2))

        # Gradient computation: dJ/dw and dJ/db
        dw = float((1 / m) * np.sum((y_pred - y) * X_scaled))
        db = float((1 / m) * np.sum(y_pred - y))

        # Weight updates
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            history.append({
                "epoch": epoch,
                "loss": round(loss, 4),
                "w_scaled": round(w, 4),
                "b": round(b, 4)
            })

    # Unscale weight for original equation: y = w_orig * X + b_orig
    w_unscaled = w / x_std
    b_unscaled = b - (w * x_mean / x_std)

    return {
        "learning_rate": learning_rate,
        "total_epochs": epochs,
        "final_loss": round(history[-1]['loss'], 4),
        "fitted_equation": f"CGPA = {round(w_unscaled, 3)} * StudyHours + {round(b_unscaled, 2)}",
        "w": round(w_unscaled, 4),
        "b": round(b_unscaled, 4),
        "history": history
    }

# ─── 5. Chain Rule & Backpropagation Walkthrough ─────────────────────────────

def compute_chain_rule_step(x_val=2.0, target=1.0, w1=0.5, b1=0.1, w2=0.8, b2=-0.2):
    """
    Demonstrates Backpropagation and Chain Rule on a 2-Layer Neural Network node:
    z1 = w1*x + b1
    a1 = relu(z1) or sigmoid(z1)
    z2 = w2*a1 + b2
    y_pred = sigmoid(z2)
    Loss = 0.5 * (target - y_pred)^2
    """
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def d_sigmoid(sig):
        return sig * (1.0 - sig)

    # 1. Forward Pass
    z1 = w1 * x_val + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    y_pred = sigmoid(z2)
    loss = 0.5 * (target - y_pred) ** 2

    # 2. Backward Pass via Chain Rule
    # dLoss/dy_pred
    dL_dypred = -(target - y_pred)

    # dypred/dz2
    dypred_dz2 = d_sigmoid(y_pred)

    # dLoss/dz2 = (dLoss/dy_pred) * (dypred/dz2)
    dL_dz2 = dL_dypred * dypred_dz2

    # dLoss/dw2 = dL_dz2 * (dz2/dw2) = dL_dz2 * a1
    dL_dw2 = dL_dz2 * a1

    # dLoss/db2 = dL_dz2 * (dz2/db2) = dL_dz2 * 1
    dL_db2 = dL_dz2 * 1.0

    # Backpropagate to layer 1:
    # dLoss/da1 = dL_dz2 * (dz2/da1) = dL_dz2 * w2
    dL_da1 = dL_dz2 * w2

    # da1/dz1
    da1_dz1 = d_sigmoid(a1)

    # dLoss/dz1 = dL_da1 * da1_dz1
    dL_dz1 = dL_da1 * da1_dz1

    # dLoss/dw1 = dL_dz1 * (dz1/dw1) = dL_dz1 * x
    dL_dw1 = dL_dz1 * x_val

    # dLoss/db1 = dL_dz1 * 1
    dL_db1 = dL_dz1 * 1.0

    return {
        "inputs": {"x": float(x_val), "target": float(target)},
        "forward": {
            "z1": round(float(z1), 4),
            "a1": round(float(a1), 4),
            "z2": round(float(z2), 4),
            "y_pred": round(float(y_pred), 4),
            "loss": round(float(loss), 6)
        },
        "chain_rule_gradients": {
            "dL_dw2": round(float(dL_dw2), 6),
            "dL_db2": round(float(dL_db2), 6),
            "dL_dw1": round(float(dL_dw1), 6),
            "dL_db1": round(float(dL_db1), 6)
        }
    }

if __name__ == '__main__':
    print("[WEEK 5-6] Testing Math Engine...")
    v_res = vector_operations([3, 4], [1, 2])
    print("Vector ops:", v_res)
    e_res = compute_eigen_analysis()
    print("Eigenvalues:", e_res['eigenvalues'])
    gd_res = run_gradient_descent(0.05, 50)
    print("Gradient descent fitted:", gd_res['fitted_equation'])
    cr_res = compute_chain_rule_step()
    print("Chain rule gradients:", cr_res['chain_rule_gradients'])
