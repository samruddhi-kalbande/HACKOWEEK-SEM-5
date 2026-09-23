"""Week 11–12 practical: PCA and t-SNE on the Week 9–10 student dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "WEEK9 - WEEK10" / "data" / "kaggle_student_performance.csv"
FEATURES = [
    "CGPA",
    "AttendanceRate",
    "StudyHoursPerWeek",
    "MathScore",
    "ReadingScore",
    "WritingScore",
]


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # Keep the same six numeric features used for the previous clustering work.
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # Standardization is important because the features use different ranges.
    X_scaled = StandardScaler().fit_transform(X)
    df["Distinction"] = (df["CGPA"] >= 8.5).astype(int)

    # PCA: inspect all components, then use the first two for visualization.
    pca_all = PCA().fit(X_scaled)
    explained = pca_all.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    print("PCA explained variance ratio:", np.round(explained, 4))
    print("PCA cumulative explained variance:", np.round(cumulative, 4))

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_scaled)
    print(f"First two PCA components retain {pca.explained_variance_ratio_.sum():.2%} of variance.")

    pca_frame = pd.DataFrame({"PC1": pca_2d[:, 0], "PC2": pca_2d[:, 1], "Distinction": df["Distinction"]})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_frame, x="PC1", y="PC2", hue="Distinction", palette="Set1", s=100)
    plt.title("PCA Projection of Student Performance")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "pca_projection.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    components = np.arange(1, len(explained) + 1)
    plt.plot(components, explained, marker="o", label="Individual variance")
    plt.plot(components, cumulative, marker="s", label="Cumulative variance")
    plt.xticks(components)
    plt.ylim(0, 1.05)
    plt.xlabel("Number of principal components")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "pca_explained_variance.png", dpi=160)
    plt.close()

    # t-SNE is local and exploratory. With this small dataset, perplexity must be < n_samples.
    perplexity = min(5, len(X_scaled) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
    )
    tsne_2d = tsne.fit_transform(X_scaled)
    tsne_frame = pd.DataFrame({"tSNE1": tsne_2d[:, 0], "tSNE2": tsne_2d[:, 1], "Distinction": df["Distinction"]})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tsne_frame, x="tSNE1", y="tSNE2", hue="Distinction", palette="Set1", s=100)
    plt.title(f"t-SNE Visualization of Student Performance (perplexity={perplexity})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "tsne_projection.png", dpi=160)
    plt.close()

    print(f"Saved PCA and t-SNE plots to: {BASE_DIR}")


if __name__ == "__main__":
    main()
