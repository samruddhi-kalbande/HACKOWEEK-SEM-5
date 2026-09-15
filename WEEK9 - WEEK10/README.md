# Week 9 – Week 10: Complete Scikit-Learn Workflow & Clustering

[![Scikit-Learn](https://img.shields.io/badge/Workflow-Scikit--Learn%20Pipelines-orange.svg)](https://scikit-learn.org/)
[![Evaluation](https://img.shields.io/badge/Evaluation-ROC--AUC%20%7C%20Cross--Validation-green.svg)](https://scikit-learn.org/stable/modules/model_evaluation.html)
[![Clustering](https://img.shields.io/badge/Clustering-K--Means%20%7C%20Hierarchical%20%7C%20DBSCAN-blue.svg)](https://scikit-learn.org/stable/modules/clustering.html)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week9_week10_ml_pipeline_clustering.ipynb)

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Scikit-Learn Workflow**:
  - Preprocessing Pipelines: Numerical median imputation (`SimpleImputer`) and scaling (`StandardScaler`), categorical one-hot encoding (`OneHotEncoder`) with `ColumnTransformer`.
  - Feature engineering and pipeline integration.
- **Model Evaluation**:
  - Stratified Train/Test Split
  - K-Fold Cross-Validation (`cross_val_score`)
  - Confusion Matrix (TP, TN, FP, FN), Precision, Recall, Specificity, F1-Score
  - ROC Curve (Receiver Operating Characteristic) and ROC-AUC metric calculation
- **Unsupervised Clustering**:
  - K-Means Clustering (Centroid-based, inertia, silhouette score)
  - Hierarchical Agglomerative Clustering (Ward linkage)
  - DBSCAN (Density-Based Spatial Clustering with noise detection)
  - 2D PCA visual projection of student behavioral cohorts.

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough, code outputs, confusion matrices, and ROC curves are in:
[`notebooks/week9_week10_ml_pipeline_clustering.ipynb`](notebooks/week9_week10_ml_pipeline_clustering.ipynb)

To launch the notebook:
```bash
cd "WEEK9 - WEEK10"
jupyter notebook notebooks/week9_week10_ml_pipeline_clustering.ipynb
```

---

## 🌐 Interactive ML Pipeline & Clustering Studio
An interactive web dashboard has been developed allowing users to:
1. **⚡ Scikit-Learn Pipeline Visualizer**: Inspect the complete ColumnTransformer and Logistic Regression pipeline, viewing hold-out test metrics and cross-validation fold accuracies.
2. **📈 ROC-AUC & Evaluation**: Interactive Confusion Matrix breakdown and HTML5 Canvas ROC curve rendering with real-time AUC score display.
3. **🧬 Unsupervised Clustering Studio**: Interactively run K-Means, Hierarchical, or DBSCAN clustering; adjust cluster count $k$ or epsilon $\epsilon$; render a live 2D PCA scatter plot; and inspect student cohort demographic profiles (High Achievers, Balanced, At-Risk).

### How to Run the Web Dashboard:
```bash
cd "WEEK9 - WEEK10"
python app.py
```
Then navigate to: **`http://localhost:5004`**
