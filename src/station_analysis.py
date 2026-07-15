"""Per-station demand analysis for the proposed charging stations.

For each proposed station this module computes: the number of assigned
EVs, their mean and minimum electric range, the mean and maximum
distance from the EVs to the station, and an estimate of the number of
charging ports needed.

This replaces the original MATLAB scripts (SCfun.m, SCinfo.m,
rangeplot.m). Because every EV already carries a `station_id` from
`clustering.py`, no manual index bookkeeping between files is needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Approximate miles per degree at Washington State's latitude,
# used to convert coordinate differences into distances.
MILES_PER_DEG_LON = 54.6
MILES_PER_DEG_LAT = 69.0

# Charging port estimate coefficients (see the project report for the
# derivation of the demand assumptions).
DEMAND_FACTORS = 0.34 * 0.5 * 0.3
PORT_CAPACITY = 20 * 3


def distances_to_station(evs: pd.DataFrame, station: pd.Series) -> np.ndarray:
    """Distance in miles from each EV to its station centroid."""
    dx = (evs["longitude"] - station["longitude"]) * MILES_PER_DEG_LON
    dy = (evs["latitude"] - station["latitude"]) * MILES_PER_DEG_LAT
    return np.sqrt(dx**2 + dy**2)


def estimate_ports(n_ev: int, mean_range: float) -> float:
    """Estimate the number of charging ports needed at a station."""
    return (n_ev * mean_range * DEMAND_FACTORS) / PORT_CAPACITY


def station_stats(assignments: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Compute per-station demand statistics."""
    rows = []
    for _, station in stations.iterrows():
        evs = assignments[assignments["station_id"] == station["station_id"]]
        dist = distances_to_station(evs, station)
        rows.append({
            "station_id": int(station["station_id"]),
            "n_ev": len(evs),
            "mean_range_mi": evs["Electric Range"].mean(),
            "min_range_mi": evs["Electric Range"].min(),
            "mean_dist_mi": dist.mean(),
            "max_dist_mi": dist.max(),
            "est_ports": estimate_ports(len(evs), evs["Electric Range"].mean()),
        })
    return pd.DataFrame(rows)


def plot_range_vs_distance(stats: pd.DataFrame, out_path: Path) -> None:
    """Check that station spacing is feasible for the EVs assigned to it."""
    plt.figure()
    plt.plot(stats["station_id"], stats["min_range_mi"], linewidth=2, label="Min EV range")
    plt.plot(stats["station_id"], stats["max_dist_mi"], linewidth=2, label="Max EV-to-station distance")
    plt.grid(True)
    plt.xlabel("Charging station area")
    plt.ylabel("Miles")
    plt.title("Min EV range vs max EV-to-station distance")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ev_and_ports(stats: pd.DataFrame, out_path: Path) -> None:
    plt.figure()
    plt.plot(stats["station_id"], stats["n_ev"], linewidth=2, label="Number of EVs")
    plt.plot(stats["station_id"], stats["est_ports"], linewidth=2, label="Estimated charging ports")
    plt.grid(True)
    plt.xlabel("Charging station area")
    plt.ylabel("Count")
    plt.title("EVs and estimated charging ports per station area")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = root / "results"

    assignments = pd.read_excel(results / "ev_station_assignments.xlsx")
    stations = pd.read_excel(results / "stations.xlsx")

    stats = station_stats(assignments, stations)
    stats.to_excel(results / "station_stats.xlsx", index=False)
    plot_range_vs_distance(stats, results / "range_vs_distance.png")
    plot_ev_and_ports(stats, results / "ev_and_ports.png")
    print(stats.round(2).to_string(index=False))
