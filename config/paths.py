"""
Paths Configuration for Financial ML Training Pipeline
Author: Marcos López de Prado AFML Framework
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input data paths
DEFAULT_DATASET_PATH = Path(
    r"C:\Users\profe\OneDrive\Documents\cAlgo\Sources\Robots\X_Cusum\X_CUSUM_Exports\USDJPY\USDJPY_DataSetHistory.parquet"
)
AFML_PARTS_DIR = Path(r"C:\Users\profe\Downloads\AFML_Parts")
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

# Outputs directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "trained_models"
PLOTS_DIR = OUTPUTS_DIR / "feature_plots"
LOGS_DIR = OUTPUTS_DIR / "logs"

# cTrader cBot integration directory
CBOT_PROJECT_DIR = Path(
    r"C:\Users\profe\OneDrive\Documents\cAlgo\Sources\Robots\Mission learning bot version 1\Mission learning bot version 1"
)
CBOT_SLN_PATH = Path(
    r"C:\Users\profe\OneDrive\Documents\cAlgo\Sources\Robots\Mission learning bot version 1\Mission learning bot version 1.sln"
)
