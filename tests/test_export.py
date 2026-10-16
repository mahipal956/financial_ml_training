"""
Unit Tests for Model Exporters
Covers: ONNX pipeline serialization, ONNXRuntime inference, and JSON configuration exporter.
"""
import unittest
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from models.pipeline import MyPipeline
from export.onnx_exporter import export_model_to_onnx
from export.json_exporter import export_config_to_json
import onnxruntime as ort


class TestExport(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tmp_dir = Path("outputs/test_export")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = self.tmp_dir / "test_model.onnx"
        self.json_path = self.tmp_dir / "test_config.json"

    def tearDown(self):
        if self.onnx_path.exists():
            self.onnx_path.unlink()
        if self.json_path.exists():
            self.json_path.unlink()

    def test_onnx_export_and_inference(self):
        clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=5, random_state=42)
        pipe = MyPipeline([("model", clf)])
        X = np.random.randn(40, 4).astype(np.float32)
        y = np.random.randint(0, 2, 40)
        pipe.fit(X, y)

        success = export_model_to_onnx(pipe, n_features=4, output_path=self.onnx_path)
        self.assertTrue(success)
        self.assertTrue(self.onnx_path.exists())

        # Test ONNX Runtime inference
        session = ort.InferenceSession(str(self.onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_preds = session.run(None, {input_name: X[:5]})
        self.assertGreater(len(onnx_preds), 0)

    def test_json_config_export(self):
        features = ["f0", "f1", "f2", "f3"]
        stats = {
            f: {"mean": 0.0, "std": 1.0, "min": -2.0, "max": 2.0}
            for f in features
        }
        success = export_config_to_json(
            feature_names=features,
            step_size=0.05,
            execution_threshold=0.55,
            feature_stats=stats,
            model_params={"avg_u": 0.85, "cv_f1_mean": 0.72},
            output_path=self.json_path,
        )
        self.assertTrue(success)
        self.assertTrue(self.json_path.exists())

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.assertEqual(data["num_features"], 4)
        self.assertEqual(data["step_size"], 0.05)
        self.assertEqual(data["execution_threshold"], 0.55)
        self.assertIn("f0", data["feature_stats"])
        self.assertEqual(data["model_params"]["avg_u"], 0.85)


if __name__ == "__main__":
    unittest.main()
