import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import logging

from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_tpr_threshold(
    yval: np.ndarray, ypred: np.ndarray, target_tnr: float
) -> Tuple[float, pd.DataFrame]:
    probas = pd.DataFrame({
        "yval": yval,
        "ypred": ypred
    })

    negative_probas = probas[probas['yval'] == 0]
    threshold = negative_probas['ypred'].quantile(target_tnr)
    logger.info(f"Percentile TNR (80th percentile): {threshold:.6f}")
    return threshold, probas