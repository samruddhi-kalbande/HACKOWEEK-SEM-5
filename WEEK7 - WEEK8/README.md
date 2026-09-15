# Week 7 – Week 8: Supervised Machine Learning (Regression & Classification)

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Regression](https://img.shields.io/badge/Regression-Linear%20%7C%20Poly%20%7C%20Ridge%20%7C%20Lasso-blue.svg)](https://scikit-learn.org/stable/modules/linear_model.html)
[![Classification](https://img.shields.io/badge/Classification-Logistic%20%7C%20KNN-green.svg)](https://scikit-learn.org/stable/modules/neighbors.html)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week7_week8_regression_classification.ipynb)

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Regression**:
  - Linear Regression (Ordinary Least Squares)
  - Polynomial Regression (Degree 2 with interactions)
  - Ridge Regression ($L_2$ Regularization)
  - Lasso Regression ($L_1$ Regularization & Feature Selection)
- **Classification**:
  - Logistic Regression (Sigmoid activation, decision thresholding)
  - K-Nearest Neighbors (KNN Classification with distance metrics)

---

## 📓 Jupyter Notebook
The complete model comparisons, formulas, confusion matrices, and code cells are in:
[`notebooks/week7_week8_regression_classification.ipynb`](notebooks/week7_week8_regression_classification.ipynb)

To launch the notebook:
```bash
cd "WEEK7 - WEEK8"
jupyter notebook notebooks/week7_week8_regression_classification.ipynb
```

---

## 🌐 Interactive Supervised ML Studio
An interactive web dashboard has been developed allowing users to:
1. **📊 Model Benchmarks**: Comparative evaluation scorecard ($R^2$, MSE, Accuracy, F1) across all 6 models.
2. **🎯 Live Student Predictor**: Move sliders for study hours, attendance, and exam marks to receive real-time regression forecasts (Linear, Poly, Ridge, Lasso) and classification distinction probabilities.
3. **🎛️ Hyperparameter Tuning**: Adjust Ridge $\alpha$, Lasso $\alpha$, and KNN $k$ in real time and watch model weights and benchmarks update.

### How to Run the Web Dashboard:
```bash
cd "WEEK7 - WEEK8"
python app.py
```
Then navigate to: **`http://localhost:5003`**
