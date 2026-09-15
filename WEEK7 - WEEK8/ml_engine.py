import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')

FEATURE_COLS = ['StudyHoursPerWeek', 'AttendanceRate', 'MathScore', 'ReadingScore', 'WritingScore']

def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_reg = df['CGPA'].to_numpy(dtype=float)
    
    # Classification target: High Academic Standing (CGPA >= 8.5)
    y_clf = (df['CGPA'] >= 8.5).astype(int).to_numpy()
    
    return df, X, y_reg, y_clf

def train_and_benchmark(ridge_alpha=1.0, lasso_alpha=0.05, knn_k=3):
    df, X, y_reg, y_clf = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_scaled, y_reg)
    y_pred_lr = lr.predict(X_scaled)
    r2_lr = r2_score(y_reg, y_pred_lr)
    mse_lr = mean_squared_error(y_reg, y_pred_lr)

    # 2. Polynomial Regression (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y_reg)
    y_pred_poly = poly_reg.predict(X_poly)
    r2_poly = r2_score(y_reg, y_pred_poly)
    mse_poly = mean_squared_error(y_reg, y_pred_poly)

    # 3. Ridge Regression (L2)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_scaled, y_reg)
    y_pred_ridge = ridge.predict(X_scaled)
    r2_ridge = r2_score(y_reg, y_pred_ridge)
    mse_ridge = mean_squared_error(y_reg, y_pred_ridge)

    # 4. Lasso Regression (L1)
    lasso = Lasso(alpha=lasso_alpha, max_iter=2000)
    lasso.fit(X_scaled, y_reg)
    y_pred_lasso = lasso.predict(X_scaled)
    r2_lasso = r2_score(y_reg, y_pred_lasso)
    mse_lasso = mean_squared_error(y_reg, y_pred_lasso)

    # 5. Logistic Regression
    clf_log = LogisticRegression()
    clf_log.fit(X_scaled, y_clf)
    y_pred_log = clf_log.predict(X_scaled)
    acc_log = accuracy_score(y_clf, y_pred_log)
    f1_log = f1_score(y_clf, y_pred_log, zero_division=0)

    # 6. K-Nearest Neighbors Classifier
    clf_knn = KNeighborsClassifier(n_neighbors=knn_k)
    clf_knn.fit(X_scaled, y_clf)
    y_pred_knn = clf_knn.predict(X_scaled)
    acc_knn = accuracy_score(y_clf, y_pred_knn)
    f1_knn = f1_score(y_clf, y_pred_knn, zero_division=0)

    return {
        "regression": {
            "linear": {
                "name": "Linear Regression (OLS)",
                "r2": round(float(r2_lr), 4),
                "mse": round(float(mse_lr), 4),
                "weights": {col: round(float(w), 4) for col, w in zip(FEATURE_COLS, lr.coef_)},
                "intercept": round(float(lr.intercept_), 4)
            },
            "polynomial": {
                "name": "Polynomial Regression (Deg 2)",
                "r2": round(float(r2_poly), 4),
                "mse": round(float(mse_poly), 4),
                "features_count": X_poly.shape[1]
            },
            "ridge": {
                "name": f"Ridge Regression (L2, alpha={ridge_alpha})",
                "r2": round(float(r2_ridge), 4),
                "mse": round(float(mse_ridge), 4),
                "weights": {col: round(float(w), 4) for col, w in zip(FEATURE_COLS, ridge.coef_)},
                "alpha": ridge_alpha
            },
            "lasso": {
                "name": f"Lasso Regression (L1, alpha={lasso_alpha})",
                "r2": round(float(r2_lasso), 4),
                "mse": round(float(mse_lasso), 4),
                "weights": {col: round(float(w), 4) for col, w in zip(FEATURE_COLS, lasso.coef_)},
                "zeroed_features": [col for col, w in zip(FEATURE_COLS, lasso.coef_) if abs(w) < 1e-4],
                "alpha": lasso_alpha
            }
        },
        "classification": {
            "logistic": {
                "name": "Logistic Regression",
                "accuracy": round(float(acc_log) * 100, 1),
                "f1_score": round(float(f1_log), 3),
                "coefficients": {col: round(float(c), 4) for col, c in zip(FEATURE_COLS, clf_log.coef_[0])}
            },
            "knn": {
                "name": f"K-Nearest Neighbors (k={knn_k})",
                "accuracy": round(float(acc_knn) * 100, 1),
                "f1_score": round(float(f1_knn), 3),
                "k": knn_k
            }
        }
    }

def predict_custom_student(study_hours, attendance, math, reading, writing, ridge_alpha=1.0, lasso_alpha=0.05, knn_k=3):
    df, X, y_reg, y_clf = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Transform input
    input_raw = np.array([[study_hours, attendance, math, reading, writing]], dtype=float)
    input_scaled = scaler.transform(input_raw)

    # Models
    lr = LinearRegression().fit(X_scaled, y_reg)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    poly_reg = LinearRegression().fit(X_poly, y_reg)
    ridge = Ridge(alpha=ridge_alpha).fit(X_scaled, y_reg)
    lasso = Lasso(alpha=lasso_alpha, max_iter=2000).fit(X_scaled, y_reg)
    clf_log = LogisticRegression().fit(X_scaled, y_clf)
    clf_knn = KNeighborsClassifier(n_neighbors=knn_k).fit(X_scaled, y_clf)

    # Predict
    pred_lr = float(np.clip(lr.predict(input_scaled)[0], 0.0, 10.0))
    pred_poly = float(np.clip(poly_reg.predict(poly.transform(input_scaled))[0], 0.0, 10.0))
    pred_ridge = float(np.clip(ridge.predict(input_scaled)[0], 0.0, 10.0))
    pred_lasso = float(np.clip(lasso.predict(input_scaled)[0], 0.0, 10.0))

    prob_log = float(clf_log.predict_proba(input_scaled)[0][1])
    pred_knn_class = int(clf_knn.predict(input_scaled)[0])
    knn_distances, knn_indices = clf_knn.kneighbors(input_scaled)

    nearest_students = df.iloc[knn_indices[0]][['StudentID', 'Name', 'Department', 'CGPA']].to_dict(orient='records')

    return {
        "inputs": {
            "study_hours": study_hours,
            "attendance": attendance,
            "math": math,
            "reading": reading,
            "writing": writing
        },
        "regression_predictions": {
            "linear_cgpa": round(pred_lr, 2),
            "polynomial_cgpa": round(pred_poly, 2),
            "ridge_cgpa": round(pred_ridge, 2),
            "lasso_cgpa": round(pred_lasso, 2),
            "consensus_average": round(float(np.mean([pred_lr, pred_poly, pred_ridge, pred_lasso])), 2)
        },
        "classification_prediction": {
            "high_performer_probability": round(prob_log * 100, 1),
            "logistic_verdict": "Distinction / High Performer" if prob_log >= 0.5 else "Standard Standing",
            "knn_verdict": "Distinction / High Performer" if pred_knn_class == 1 else "Standard Standing",
            "nearest_neighbors": nearest_students
        }
    }

if __name__ == '__main__':
    print("[WEEK 7-8] Running ML Benchmark Engine...")
    bm = train_and_benchmark()
    print("Linear Regression R2:", bm['regression']['linear']['r2'])
    print("Polynomial Regression R2:", bm['regression']['polynomial']['r2'])
    print("Ridge Regression R2:", bm['regression']['ridge']['r2'])
    print("Lasso Regression R2:", bm['regression']['lasso']['r2'])
    print("Logistic Regression Accuracy:", bm['classification']['logistic']['accuracy'])
    print("KNN Accuracy:", bm['classification']['knn']['accuracy'])
    
    pred = predict_custom_student(18.0, 92.0, 90, 88, 91)
    print("Predicted CGPA (Linear):", pred['regression_predictions']['linear_cgpa'])
    print("High Performer Probability:", pred['classification_prediction']['high_performer_probability'], "%")
