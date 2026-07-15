"""Preprocessing for the Washington State EV population dataset.

Loads the raw EV registration data, keeps battery electric vehicles (BEV),
parses registration coordinates, and clips the data to Washington State.

Data source: Washington State Department of Licensing, Electric Vehicle
Population Data (public dataset).
"""

from pathlib import Path

import pandas as pd

EV_DATA_URL = (
    "https://raw.githubusercontent.com/zhangqc00/6880Data/main/"
    "Electric_Vehicle_Population_Data.csv"
)

# Bounding limits of Washington State used to remove out-of-state samples.
WA_LON_MAX = -117.0256
WA_LAT_MIN = 45.5821

COLUMNS_OF_INTEREST = [
    "Vehicle Location",
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Electric Range",
    "Base MSRP",
]


def load_raw_data(source: str = EV_DATA_URL) -> pd.DataFrame:
    """Load the raw EV population CSV from a URL or local path."""
    return pd.read_csv(source)


def filter_bev(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only battery electric vehicles.

    Plug-in hybrids are removed because their low electric range makes
    them poor indicators of charging-station demand.
    """
    df = df[COLUMNS_OF_INTEREST]
    df = df[df["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)"]
    return df.drop(columns=["Electric Vehicle Type"]).dropna().reset_index(drop=True)


def parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Extract longitude/latitude from the 'Vehicle Location' WKT string.

    The column has the form 'POINT (-122.30839 47.610365)'.
    """
    coords = df["Vehicle Location"].str.extract(
        r"POINT \((?P<longitude>-?\d+\.?\d*) (?P<latitude>-?\d+\.?\d*)\)"
    )
    df = df.copy()
    df["longitude"] = coords["longitude"].astype(float)
    df["latitude"] = coords["latitude"].astype(float)
    return df.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)


def clip_to_washington(df: pd.DataFrame) -> pd.DataFrame:
    """Remove samples registered outside Washington State."""
    in_state = (df["longitude"] < WA_LON_MAX) & (df["latitude"] > WA_LAT_MIN)
    return df[in_state].reset_index(drop=True)


def preprocess(source: str = EV_DATA_URL) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the cleaned dataset."""
    df = load_raw_data(source)
    df = filter_bev(df)
    df = parse_coordinates(df)
    df = clip_to_washington(df)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[1] / "data" / "EVprocessed_data.xlsx"
    out_path.parent.mkdir(exist_ok=True)
    processed = preprocess()
    processed.to_excel(out_path, index=False)
    print(f"Saved {len(processed)} BEV records to {out_path}")
