import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')

NUMERIC_FEATURES = ['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'MathScore', 'ReadingScore', 'WritingScore']
CATEGORICAL_FEATURES = ['Gender', 'Department']
CLUSTERING_FEATURES = ['StudyHoursPerWeek', 'AttendanceRate', 'MathScore', 'ReadingScore', 'WritingScore', 'CGPA']

def load_dataset():
    df = pd.read_csv(DATA_FILE)
    # Target: Distinction (CGPA >= 8.5)
    df['Distinction'] = (df['CGPA'] >= 8.5).astype(int)
    return df

# ─── 1. Scikit-Learn Pipeline & Evaluation ───────────────────────────────────

def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    return full_pipeline

def evaluate_pipeline():
    df = load_dataset()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Distinction']

    pipeline = build_pipeline()

    # 1. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics on Test Set
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Confusion Matrix on full dataset for comprehensive inspection
    pipeline.fit(X, y)
    y_full_pred = pipeline.predict(X)
    y_full_prob = pipeline.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_full_pred)
    tn, fp, fn, tp = cm.ravel()

    # ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y, y_full_prob)
    auc = float(roc_auc_score(y, y_full_prob))

    # 2. Stratified 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    return {
        "pipeline_structure": [
            {"step": "ColumnTransformer", "description": "Numeric imputation + scaling, Categorical one-hot encoding"},
            {"step": "LogisticRegression", "description": "Classification estimator with L2 regularization"}
        ],
        "test_metrics": {
            "accuracy": round(acc * 100, 1),
            "precision": round(prec * 100, 1),
            "recall": round(rec * 100, 1),
            "f1_score": round(f1, 3)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "cross_validation": {
            "folds": [round(float(s) * 100, 1) for s in cv_scores],
            "mean_accuracy": round(float(np.mean(cv_scores)) * 100, 1),
            "std_dev": round(float(np.std(cv_scores)) * 100, 1)
        },
        "roc_curve": {
            "auc_score": round(auc, 3),
            "fpr": [round(float(x), 3) for x in fpr],
            "tpr": [round(float(y), 3) for y in tpr]
        }
    }

# ─── 2. Clustering Suite (K-Means, Hierarchical, DBSCAN) ─────────────────────

def run_clustering(algorithm="kmeans", k=3, eps=1.5, min_samples=2):
    df = load_dataset()
    X = df[CLUSTERING_FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2D PCA for visual projection
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X_scaled)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia = float(model.inertia_)
    elif algorithm == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X_scaled)
        inertia = None
    elif algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        inertia = None
    else:
        raise ValueError("Unknown algorithm")

    # Silhouette score (only if > 1 cluster and not all points in same cluster)
    unique_labels = set(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(labels):
        sil = float(silhouette_score(X_scaled, labels))
    else:
        sil = 0.0

    # Assemble student cluster points for interactive scatter
    points = []
    for i, row in df.iterrows():
        points.append({
            "id": row['StudentID'],
            "name": row['Name'],
            "department": row['Department'],
            "cgpa": float(row['CGPA']),
            "attendance": float(row['AttendanceRate']),
            "study_hours": float(row['StudyHoursPerWeek']),
            "cluster": int(labels[i]),
            "x": round(float(coords_2d[i, 0]), 3),
            "y": round(float(coords_2d[i, 1]), 3)
        })

    # Cluster profiles
    profiles = {}
    for c_id in unique_labels:
        mask = (labels == c_id)
        subset = df[mask]
        label_name = f"Cluster {c_id}" if c_id != -1 else "Noise / Outliers"
        profiles[str(c_id)] = {
            "name": label_name,
            "count": int(np.sum(mask)),
            "avg_cgpa": round(float(subset['CGPA'].mean()), 2) if len(subset) else 0,
            "avg_attendance": round(float(subset['AttendanceRate'].mean()), 1) if len(subset) else 0,
            "avg_study_hours": round(float(subset['StudyHoursPerWeek'].mean()), 1) if len(subset) else 0
        }

    return {
        "algorithm": algorithm,
        "cluster_count": len(unique_labels),
        "silhouette_score": round(sil, 3),
        "inertia": round(inertia, 2) if inertia else None,
        "pca_variance_ratio": [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_],
        "points": points,
        "cluster_profiles": profiles
    }

if __name__ == '__main__':
    print("[WEEK 9-10] Testing Workflow Engine...")
    eval_res = evaluate_pipeline()
    print("Cross-Val Mean Accuracy:", eval_res['cross_validation']['mean_accuracy'], "%")
    print("ROC-AUC:", eval_res['roc_curve']['auc_score'])
    
    km_res = run_clustering("kmeans", k=3)
    print("K-Means Silhouette Score:", km_res['silhouette_score'])
    
    h_res = run_clustering("hierarchical", k=3)
    print("Hierarchical Cluster count:", h_res['cluster_count'])
    
    db_res = run_clustering("dbscan", eps=1.8, min_samples=2)
    print("DBSCAN Cluster profiles:", db_res['cluster_profiles'].keys())
