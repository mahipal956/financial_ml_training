"""
ONNX Model Exporter Module
Converts trained scikit-learn pipeline into standard ONNX format for cTrader C# bot inference.
"""
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(model, n_features, output_path):
    """
    Serializes a fitted scikit-learn model or pipeline to ONNX format.
    - model: fitted estimator or pipeline.
    - n_features: number of input features.
    - output_path: path to output .onnx file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # If MyPipeline was wrapped, convert to standard Pipeline for skl2onnx
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    model_to_convert = model
    if hasattr(model, "steps"):
        from sklearn.pipeline import Pipeline
        model_to_convert = Pipeline(model.steps)

    try:
        onnx_model = convert_sklearn(
            model_to_convert,
            initial_types=initial_type,
            target_opset=12,
        )
        with open(out_file, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[ONNX] Successfully exported model to {out_file}")
        return True
    except Exception as e:
        print(f"[ONNX Warning] Standard ONNX conversion encountered: {e}")
        # Try converting underlying estimator if wrapped
        if hasattr(model, "steps"):
            final_est = model.steps[-1][1]
            try:
                onnx_model = convert_sklearn(
                    final_est,
                    initial_types=initial_type,
                    target_opset=12,
                )
                with open(out_file, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                print(f"[ONNX] Exported underlying estimator {type(final_est).__name__} to {out_file}")
                return True
            except Exception as e2:
                print(f"[ONNX Error] Fallback export failed: {e2}")
                return False
        return False
