"""Gaussian Mixture Model clustering, used as a comparison to K-means.

Model selection uses the silhouette score over a range of component
counts, followed by fitting the selected model. In the project, GMM
results were compared against the K-means pipeline and K-means was
selected for the final station placement (see report, Section on
model comparison).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.mixture import GaussianMixture

SEED = 42
FEATURES = ["longitude", "latitude"]


def silhouette_scores(points: pd.DataFrame, k_range=range(2, 11)) -> dict:
    """Silhouette score of a GMM fit for each number of components."""
    scores = {}
    for k in k_range:
        model = GaussianMixture(n_components=k, n_init=5, init_params="kmeans",
                                random_state=SEED)
        labels = model.fit_predict(points)
        scores[k] = metrics.silhouette_score(points, labels, metric="euclidean")
    return scores


def plot_silhouette(scores: dict, out_path: Path) -> None:
    plt.figure()
    plt.plot(list(scores.keys()), list(scores.values()), "bo-")
    plt.xlabel("Number of components")
    plt.ylabel("Silhouette score")
    plt.title("GMM model selection by silhouette score")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fit_gmm(points: pd.DataFrame, n_components: int) -> GaussianMixture:
    model = GaussianMixture(n_components=n_components, covariance_type="full",
                            init_params="kmeans", random_state=SEED)
    model.fit(points)
    return model


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(exist_ok=True)

    data = pd.read_excel(root / "data" / "EVprocessed_data.xlsx")
    data = data.rename(columns={"longitudinal": "longitude", "lateral": "latitude"})

    scores = silhouette_scores(data[FEATURES])
    plot_silhouette(scores, results / "gmm_silhouette.png")

    best_k = max(scores, key=scores.get)
    model = fit_gmm(data[FEATURES], best_k)
    print(f"Best k by silhouette score: {best_k}")
    print("Component means (candidate station regions):")
    print(model.means_)
