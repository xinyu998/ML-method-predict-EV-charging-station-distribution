"""Hierarchical K-means clustering of the WA electric vehicle population.

The EV registrations are clustered in two stages to propose charging
station locations:

1. A top-level K-means (k=6, chosen with the elbow method) splits the
   state into regions.
2. Each region is clustered again to find station areas. The densest
   region (the Puget Sound metro area, cluster 0) is split one level
   deeper because a single pass leaves its sub-clusters too large.

The final centroids are the proposed charging station locations. Each
vehicle keeps a `station_id` label for the demand analysis in
`station_analysis.py`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42  # fixed random state so results are reproducible
TOP_LEVEL_K = 6

# Number of sub-clusters for each top-level cluster, chosen from the
# elbow analysis of each region. Cluster 3 needed one extra.
SUBCLUSTER_K = {0: 5, 1: 5, 2: 5, 3: 6, 4: 5, 5: 5}

# Cluster 0 (Puget Sound area) is split one level deeper: each of its
# sub-clusters is clustered again with this k.
DEEP_SPLIT_CLUSTER = 0
DEEP_SPLIT_K = 5

FEATURES = ["longitude", "latitude"]


def fit_kmeans(points: pd.DataFrame, k: int) -> KMeans:
    """Fit K-means with k-means++ initialization and a fixed seed."""
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
    model.fit(points)
    return model


def elbow_curve(points: pd.DataFrame, k_range=range(2, 12)) -> dict:
    """Return the K-means inertia for each k, used to pick k visually."""
    return {k: fit_kmeans(points, k).inertia_ for k in k_range}


def plot_elbow(inertias: dict, out_path: Path) -> None:
    plt.figure()
    plt.plot(list(inertias.keys()), list(inertias.values()), "bx-")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow method for choosing k")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_map(points: pd.DataFrame, model: KMeans, title: str, out_path: Path) -> None:
    """Plot cluster decision regions, data points, and centroids."""
    h = 0.02  # mesh step size in degrees
    x_min, x_max = points.iloc[:, 0].min() - 1, points.iloc[:, 0].max() + 1
    y_min, y_max = points.iloc[:, 1].min() - 1, points.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=points.columns)
    regions = model.predict(mesh).reshape(xx.shape)

    plt.figure(figsize=(15, 8))
    plt.imshow(
        regions,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap="coolwarm",
        aspect="auto",
        origin="lower",
    )
    plt.plot(points.iloc[:, 0], points.iloc[:, 1], "k.", markersize=2)
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=120, linewidths=5, color="w")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def hierarchical_cluster(df: pd.DataFrame, plot_dir: Path | None = None):
    """Run the two-stage clustering and return (assignments, stations).

    assignments: the input dataframe with an added `station_id` column.
    stations:    dataframe of station centroids (station_id, longitude, latitude).
    """
    df = df.copy()
    df["station_id"] = -1
    station_rows = []
    next_station_id = 0

    top_model = fit_kmeans(df[FEATURES], TOP_LEVEL_K)
    df["region"] = top_model.labels_
    if plot_dir is not None:
        plot_cluster_map(df[FEATURES], top_model, "Top-level regions (k=6)",
                         plot_dir / "regions.png")

    for region, k in SUBCLUSTER_K.items():
        region_mask = df["region"] == region
        region_points = df.loc[region_mask, FEATURES]
        sub_model = fit_kmeans(region_points, k)
        sub_labels = pd.Series(sub_model.labels_, index=region_points.index)

        if plot_dir is not None:
            plot_cluster_map(region_points, sub_model,
                             f"Region {region} sub-clusters (k={k})",
                             plot_dir / f"region_{region}.png")

        if region == DEEP_SPLIT_CLUSTER:
            # Split each sub-cluster of the dense region one level deeper.
            for sub in range(k):
                sub_mask = sub_labels == sub
                deep_points = region_points.loc[sub_mask[sub_mask].index]
                deep_model = fit_kmeans(deep_points, DEEP_SPLIT_K)
                for centroid_idx, centroid in enumerate(deep_model.cluster_centers_):
                    member_idx = deep_points.index[deep_model.labels_ == centroid_idx]
                    df.loc[member_idx, "station_id"] = next_station_id
                    station_rows.append((next_station_id, *centroid))
                    next_station_id += 1
        else:
            for centroid_idx, centroid in enumerate(sub_model.cluster_centers_):
                member_idx = sub_labels[sub_labels == centroid_idx].index
                df.loc[member_idx, "station_id"] = next_station_id
                station_rows.append((next_station_id, *centroid))
                next_station_id += 1

    stations = pd.DataFrame(station_rows, columns=["station_id", "longitude", "latitude"])
    return df, stations


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(exist_ok=True)

    data = pd.read_excel(root / "data" / "EVprocessed_data.xlsx")
    # Support the column names used by the original course project.
    data = data.rename(columns={"longitudinal": "longitude", "lateral": "latitude"})

    plot_elbow(elbow_curve(data[FEATURES]), results / "elbow.png")
    assignments, stations = hierarchical_cluster(data, plot_dir=results)

    assignments.to_excel(results / "ev_station_assignments.xlsx", index=False)
    stations.to_excel(results / "stations.xlsx", index=False)
    print(f"Proposed {len(stations)} charging station locations "
          f"for {len(assignments)} vehicles. Results saved in {results}/")
