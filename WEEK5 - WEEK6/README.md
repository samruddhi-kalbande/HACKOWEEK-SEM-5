# Week 5 – Week 6: Mathematics for Machine Learning (Linear Algebra & Calculus)

[![Linear Algebra](https://img.shields.io/badge/Linear%20Algebra-Vectors%20%26%20Matrices-purple.svg)](https://en.wikipedia.org/wiki/Linear_algebra)
[![Calculus](https://img.shields.io/badge/Calculus-Gradients%20%26%20Chain%20Rule-blue.svg)](https://en.wikipedia.org/wiki/Differential_calculus)
[![PCA](https://img.shields.io/badge/Dimensionality%20Reduction-Eigen%20Decomposition-green.svg)](https://en.wikipedia.org/wiki/Principal_component_analysis)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebooks/week5_week6_linear_algebra_calculus.ipynb)

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Linear Algebra**: Vectors, matrices, dot product, norms, cosine similarity, eigenvalues and eigenvectors (intuition-level & PCA on the student covariance matrix).
- **Calculus**: Numerical & analytical derivatives, gradient vectors $\nabla J$, gradient descent optimization (fitting Study Hours vs CGPA on the Kaggle dataset), and the chain rule for neural network backpropagation intuition.

---

## 📓 Jupyter Notebook
The complete mathematical explanations, proofs, and executable code cells are contained in:
[`notebooks/week5_week6_linear_algebra_calculus.ipynb`](notebooks/week5_week6_linear_algebra_calculus.ipynb)

To launch the notebook:
```bash
cd "WEEK5 - WEEK6"
jupyter notebook notebooks/week5_week6_linear_algebra_calculus.ipynb
```

---

## 🌐 Interactive Math for ML Explorer
An interactive browser interface has been developed allowing users to:
1. **⚡ Vector Operations**: Live dot product, norm, angle, and cosine similarity between arbitrary vectors.
2. **📐 2D Matrix Visualizer**: Canvas visualizer rendering linear transformations on the unit square and computing determinants.
3. **🧬 Covariance Matrix & Eigenvalues (PCA)**: Interactive decomposition of Kaggle student academic marks.
4. **📉 Gradient Descent Simulator**: Configurable learning rate $\alpha$ and epochs, plotting real-time MSE loss convergence.
5. **🔗 Chain Rule & Backpropagation Trace**: Visualizing forward propagation and backpropagation step-by-step.

### How to Run the Web Dashboard:
```bash
cd "WEEK5 - WEEK6"
python app.py
```
Then open: **`http://localhost:5002`**
