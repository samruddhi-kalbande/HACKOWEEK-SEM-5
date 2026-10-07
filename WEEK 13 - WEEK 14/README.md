# Week 13 – Week 14: Ensemble Methods, Bias-Variance & Regularization

[![Ensemble](https://img.shields.io/badge/Ensemble-Bagging%20%7C%20Boosting-green.svg)](https://scikit-learn.org/stable/modules/ensemble.html)
[![XGBoost](https://img.shields.io/badge/Boosting-XGBoost%20%7C%20LightGBM-blue.svg)](https://xgboost.readthedocs.io/)
[![Regularization](https://img.shields.io/badge/Regularization-L1%20%7C%20L2-orange.svg)](https://scikit-learn.org/stable/modules/linear_model.html)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week13_week14_ensemble_regularization.ipynb)

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Ensemble Methods**:
  - Bagging: Bootstrap Aggregating, Random Forest (parallel base learners, variance reduction)
  - Boosting: Gradient Boosting, XGBoost, LightGBM (sequential error-correcting, bias reduction)
- **Bias-Variance Trade-off**:
  - Understanding model complexity vs. generalisation
  - Tree depth analysis: underfitting (high bias) → optimal → overfitting (high variance)
- **Overfitting & Underfitting**:
  - Causes, identification via learning curves, prevention strategies
  - Comparing shallow models (underfitting) vs deep models (overfitting) vs balanced models
- **Regularization (L1/L2)**:
  - L1 (Lasso): Sparse coefficients, built-in feature selection
  - L2 (Ridge): Uniform weight shrinkage, weight decay
  - C-sweep analysis (inverse regularization strength)

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough is in:
[`notebooks/week13_week14_ensemble_regularization.ipynb`](notebooks/week13_week14_ensemble_regularization.ipynb)

To launch the notebook:
```bash
cd "WEEK13 - WEEK14"
jupyter notebook notebooks/week13_week14_ensemble_regularization.ipynb
```

---

## 🌐 Interactive Ensemble & Regularization Studio
An interactive web dashboard has been developed allowing users to:
1. **🌲 Ensemble Methods**: Compare Bagging (Random Forest, BaggingClassifier) vs Boosting (GradientBoosting, XGBoost, LightGBM) with full accuracy, precision, recall, F1, ROC-AUC, and cross-validation metrics.
2. **⚖️ Bias-Variance Trade-off**: Visualise train vs test accuracy curves across tree depths, identify the optimal complexity zone, and inspect the generalisation gap.
3. **📉 Overfitting & Underfitting**: Interactive learning curves for underfit, good-fit, and overfit scenarios with causes and prevention strategies.
4. **🔒 Regularization (L1/L2)**: C-sweep comparison chart, coefficient magnitude visualisation (None vs L1 vs L2), and L1 sparsity table showing feature elimination.

### How to Run the Web Dashboard:
```bash
cd "WEEK13 - WEEK14"
pip install flask flask-cors scikit-learn pandas numpy xgboost lightgbm
python app.py
```
Then navigate to: **`http://localhost:5006`**

---

## 📂 Project Structure
```
WEEK13 - WEEK14/
├── app.py                  # Flask web server (port 5006)
├── ensemble_engine.py      # ML engine: ensemble, bias-variance, regularization
├── README.md
├── data/
│   └── kaggle_student_performance.csv
├── notebooks/
│   └── week13_week14_ensemble_regularization.ipynb
└── public/
    ├── index.html          # Interactive dashboard
    ├── css/
    │   └── styles.css      # Dark glassmorphism theme
    └── js/
        └── app.js          # Client-side rendering & API calls
```
