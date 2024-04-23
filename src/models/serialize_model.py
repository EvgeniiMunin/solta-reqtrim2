import joblib
import pickle
from sklearn.linear_model import LogisticRegression
import catboost as cb
import logging

from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def serialize_predictor(model, path: str):
    joblib.dump(model, path)


def serialize_sklearn_onnx(model, path: str, feats: list):
    initial_types = [("float_input", FloatTensorType([None, len(feats)]))]
    onnx_model = to_onnx(model, initial_types)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def serialize_onnx(model, path: str, feats: list = None):
    if isinstance(model, LogisticRegression):
        if feats is None:
            raise ValueError("Feature names must be provided for LogReg model")

        initial_types = [("float_input", FloatTensorType([None, len(feats)]))]
        onnx_model = to_onnx(model, initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Saved sklearn model to {path}")

    elif isinstance(model, cb.CatBoostClassifier):
        model.save_model(
            path,
            format="onnx",
        )
        logger.info(f"Saved CatBoost model to {path}")

    else:
        raise TypeError("Unsupported model type for serialization.")
