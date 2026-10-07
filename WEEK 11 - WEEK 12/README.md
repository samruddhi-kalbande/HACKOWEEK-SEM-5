# Week 11–12: Dimensionality Reduction with PCA and t-SNE

This practical applies **Principal Component Analysis (PCA)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)** to the student-performance dataset used in Week 9–10.

## Dataset

Source: [`WEEK9 - WEEK10/data/kaggle_student_performance.csv`](../WEEK9%20-%20WEEK10/data/kaggle_student_performance.csv)

The analysis uses the following academic and engagement features:

- `CGPA`
- `AttendanceRate`
- `StudyHoursPerWeek`
- `MathScore`
- `ReadingScore`
- `WritingScore`

Identifiers and personal fields are intentionally excluded. The numeric features are standardized before dimensionality reduction so that features with larger numeric scales do not dominate the result.

## Concepts

### PCA intuition

PCA creates new, uncorrelated axes called **principal components**. The first component captures the greatest possible variance in the data, the second captures the greatest remaining variance, and so on. PCA is useful for:

- reducing many correlated features to a smaller set;
- removing redundancy before modeling;
- visualizing high-dimensional data in two dimensions;
- measuring how much information is retained using explained variance.

PCA is a linear technique, so the resulting projection preserves global variance patterns rather than all local relationships.

### t-SNE intuition

t-SNE converts pairwise similarities into probabilities and searches for a low-dimensional map with similar neighborhood probabilities. Points that are close in the original feature space tend to remain close in the 2D visualization.

It is useful for exploring possible groups, outliers, and local structure. Unlike PCA, t-SNE is primarily a visualization method:

- distances between far-apart groups should not be over-interpreted;
- cluster sizes and spacing can change between runs;
- results depend on `perplexity`, initialization, and `random_state`;
- it should not normally be used as a production feature transformation.

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Run the analysis

From the repository root:

```bash
python "WEEK11 - WEEK12/dimensionality_reduction.py"
```

The script prints the PCA explained variance and saves these figures in this folder:

- `pca_projection.png` — students projected onto the first two principal components;
- `pca_explained_variance.png` — variance explained by each component and cumulatively;
- `tsne_projection.png` — t-SNE 2D visualization.

The plots are colored by the existing `Distinction` indicator (`CGPA >= 8.5`) only for interpretation; the label is not used to fit PCA or t-SNE.

## Interpretation

- A high cumulative explained-variance ratio for the first two or three components means the dataset can be represented compactly with limited information loss.
- PCA axes are linear combinations of the original features. Their loadings can be inspected to understand which academic or engagement variables influence each component.
- t-SNE can reveal local neighborhoods among students, but apparent clusters should be treated as exploratory rather than definitive evidence of separable student groups.

## Files

```text
WEEK11 - WEEK12/
├── README.md
├── dimensionality_reduction.py
├── pca_projection.png          # generated after running the script
├── pca_explained_variance.png  # generated after running the script
└── tsne_projection.png         # generated after running the script
```
