"""
JSON Configuration Exporter Module
Exports model metadata, feature order, thresholds, and bet-sizing parameters for cTrader bot integration.
"""
from pathlib import Path
import json


def export_config_to_json(
    feature_names,
    step_size=0.05,
    execution_threshold=0.5,
    feature_stats=None,
    model_params=None,
    output_path="outputs/trained_models/model_config.json",
):
    """
    Exports a comprehensive JSON configuration file loaded by the cTrader cBot.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "1.0",
        "description": "Financial ML Model Runtime Configuration for cTrader",
        "feature_names": list(feature_names),
        "num_features": len(feature_names),
        "execution_threshold": float(execution_threshold),
        "step_size": float(step_size),
        "feature_stats": feature_stats or {},
        "model_params": model_params or {},
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"[JSON Config] Successfully exported model configuration to {out_file}")
    return True
