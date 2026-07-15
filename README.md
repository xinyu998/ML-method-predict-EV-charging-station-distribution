# EV Charging Station Placement for Washington State

Machine learning pipeline that proposes electric vehicle (EV) charging
station locations for Washington State, based on the registration
locations and electric range of ~60,000 battery electric vehicles.

Developed as the final project for CHEME 6880 (Data Analytics and
Machine Learning) at Cornell University. I led the project: I wrote the
preprocessing, clustering, and analysis code, proposed the two-stage
clustering approach, and wrote most of the final report
(`report/6880finalreport.pdf`). The code was refactored in 2026 for
clarity and reproducibility; the methods and results are unchanged.

## Problem

Where should charging stations be placed so that every registered BEV
can reach one within its electric range, and how many charging ports
does each station need?

## Method

1. **Preprocessing** (`src/preprocessing.py`)
   Load the WA Department of Licensing EV population dataset, keep
   battery electric vehicles, parse registration coordinates from WKT
   strings, and clip to Washington State.

2. **Hierarchical K-means clustering** (`src/clustering.py`)
   A top-level K-means (k = 6, chosen with the elbow method) splits the
   state into regions. Each region is clustered again to obtain station
   areas; the dense Puget Sound region is split one level deeper. The
   final centroids (51 in total) are the proposed station locations.

3. **Model comparison** (`src/gmm_comparison.py`)
   Gaussian Mixture Models with silhouette-score model selection were
   evaluated as an alternative; K-means was selected for the final
   placement.

4. **Demand analysis** (`src/station_analysis.py`)
   For each proposed station: number of assigned EVs, mean/min electric
   range, mean/max EV-to-station distance, and an estimated number of
   charging ports. A feasibility check compares each area's minimum EV
   range against its maximum EV-to-station distance.

## Repository structure

```
├── src/
│   ├── preprocessing.py       # data loading and cleaning
│   ├── clustering.py          # two-stage K-means station placement
│   ├── gmm_comparison.py      # GMM alternative for comparison
│   └── station_analysis.py    # per-station demand statistics and plots
├── data/                      # processed dataset (generated)
├── results/                   # figures and result tables (generated)
├── report/
│   └── 6880finalreport.pdf    # full project report
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python src/preprocessing.py       # downloads and cleans the data
python src/clustering.py          # produces station locations + plots
python src/station_analysis.py    # produces demand statistics + plots
python src/gmm_comparison.py      # optional: GMM comparison
```

## Results

The pipeline proposes 51 station locations. For nearly all station
areas, the maximum EV-to-station distance is well below the minimum
electric range of the EVs in that area, indicating feasible coverage.
See `results/` after running, and the report for full discussion.
