# Data Ingestion Format

This directory contains templates and documentation for the input data required by the AFML training pipeline.

## Required CSV/Parquet Structure
To ensure compatibility with the `data_loader` module, input datasets must contain the following columns:

- `Close_Time_UTC`: Datetime index (ISO 8601).
- `Close`: Price at the close of the bar.
- `Bar_Index`: Integer index of the bar.
- `Trade_Lifespan_Bars`: Integer representing the duration of the event.
- `Primary_Model_Side`: The side of the trade (+1 or -1).
- `Meta_Label_Bin_y_Binary`: The target label (0 or 1).
- `Sample_Weight_w`: Calculated sample weight (optional).
- `Decayed_Sample_Weight`: Time-decayed sample weight (optional).

## Feature Columns
The pipeline expects the following features by default:
- `X_ATR_14`: Average True Range (14-period).
- `X_Target_Vol`: Target volatility.
- `X_Close_FFD`: Fractionally Differentiated Close price.
- `X_RSI_14`: Relative Strength Index (14-period).