# Week 3 – Week 4: Python Essentials, NumPy, Pandas & Data Visualization

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Broadcasting-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Cleaning%20%26%20GroupBy-150458.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Visualization-Matplotlib%20%26%20Seaborn-green.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebooks/week3_week4_python_numpy_pandas_viz.ipynb)

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Python Essentials**: Functions, Object-Oriented Programming (OOP with `StudentRecord`), and List/Dict comprehensions.
- **NumPy**: Multidimensional arrays, broadcasting, and vectorized operations (z-score standardization, composite scores).
- **Pandas**: DataFrames, cleaning, relational merging with course enrollments (`pd.merge`), and multi-metric `groupby` aggregations.
- **Data Visualization**: Publication-quality plots using Matplotlib and Seaborn (KDE distributions, departmental boxplots, study hours vs CGPA regression, correlation heatmap).

---

## 📓 Jupyter Notebook
The complete coding walkthrough and outputs are contained in:
[`notebooks/week3_week4_python_numpy_pandas_viz.ipynb`](notebooks/week3_week4_python_numpy_pandas_viz.ipynb)

To launch the notebook locally:
```bash
cd "WEEK3 - WEEK4"
jupyter notebook notebooks/week3_week4_python_numpy_pandas_viz.ipynb
```

---

## 🌐 Interactive Web Interface
An interactive dashboard has been developed allowing users to:
1. Explore all 4 generated Matplotlib/Seaborn visualization plots with interactive summaries.
2. Run live server-side **Pandas GroupBy** queries (select dimension, metric, and aggregation function).
3. Inspect relational data: Students, Course Enrollments, and Merged Inner-Join tables.
4. Experiment with **NumPy Vectorization & Array Broadcasting** via a live Z-score calculator.

### How to Run the Web Dashboard:
```bash
cd "WEEK3 - WEEK4"
python app.py
```
Then navigate to: **`http://localhost:5001`**
