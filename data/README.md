# Data Ingestion Format

This directory contains raw and processed data for the AFML training pipeline. 

## Required CSV/Parquet Format
To ensure compatibility with the `data_loader` module, datasets should contain the following columns:

- `Close_Time_UTC`: Datetime index (ISO 8601).
- `Close`: Price at the end of the bar.
- `X_ATR_14`, `X_Target_Vol`, `X_Close_FFD`, `X_RSI_14`: Feature set.
- `Meta_Label_Bin_y_Binary`: Target variable (0 or 1).
- `Primary_Model_Side`: Trade side (-1 or 1).
- `Trade_Lifespan_Bars`: Integer representing the duration of the event.
- `Bar_Index`: Integer index for time-alignment.
- `Sample_Weight_w`: (Optional) Weighting for training samples.

## Sample Template
Use `sample_market_data.csv` as a reference for the expected schema.