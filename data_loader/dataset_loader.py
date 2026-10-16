"""
Data Loader & Dataset Alignment Module
Ingests feature matrix X, meta-labels y, event boundaries t1, and sample weights from Parquet.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def load_dataset(parquet_path, feature_cols=None, target_col="Meta_Label_Bin_y_Binary", events_only=True):
    """
    Loads and aligns financial time series data for training.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_parquet(path)

    # Ensure datetime index
    if "Close_Time_UTC" in df.columns:
        df["Close_Time_UTC"] = pd.to_datetime(df["Close_Time_UTC"])
        df = df.set_index("Close_Time_UTC")
    df = df.sort_index()

    # Default features if not specified
    if feature_cols is None:
        feature_cols = [
            "X_ATR_14",
            "X_Target_Vol",
            "X_Close_FFD",
            "X_RSI_14",
        ]

    # Filter to active trading events where meta-label is 0 or 1
    if target_col in df.columns:
        valid_trades = df[target_col].isin([0, 1])
        if valid_trades.sum() >= 10:
            df_events = df[valid_trades].copy()
        else:
            df_events = df.copy()
    else:
        df_events = df.copy()

    # Ensure index has unique timestamps
    df_events = df_events[~df_events.index.duplicated(keep="first")].copy()

    # Construct t1 (Event end times)
    # Using Close_Time_UTC index and Trade_Lifespan_Bars
    close_times = df.index
    t1_list = []
    for idx_row in df_events.itertuples():
        bar_idx = getattr(idx_row, "Bar_Index", 0)
        lifespan = getattr(idx_row, "Trade_Lifespan_Bars", 1)
        target_bar = min(len(close_times) - 1, bar_idx + max(1, lifespan))
        t1_list.append(close_times[target_bar])

    t1 = pd.Series(t1_list, index=df_events.index, name="t1")

    # Features X
    available_features = [col for col in feature_cols if col in df_events.columns]
    X = df_events[available_features].ffill().fillna(0.0)

    # Target y
    y = df_events[target_col].astype(int)

    # Sample weights w
    if "Decayed_Sample_Weight" in df_events.columns and df_events["Decayed_Sample_Weight"].sum() > 0:
        w = df_events["Decayed_Sample_Weight"].copy()
    elif "Sample_Weight_w" in df_events.columns and df_events["Sample_Weight_w"].sum() > 0:
        w = df_events["Sample_Weight_w"].copy()
    else:
        w = pd.Series(1.0, index=df_events.index)

    # Ensure no zero weights
    w = w.clip(lower=1e-4)

    # Close series
    close = df["Close"]

    # Align indexes strictly
    common_idx = X.index.intersection(y.index).intersection(t1.index).intersection(w.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    t1 = t1.loc[common_idx]
    w = w.loc[common_idx]

    return {
        "X": X,
        "y": y,
        "t1": t1,
        "sample_weight": w,
        "close": close,
        "full_df": df_events,
    }
