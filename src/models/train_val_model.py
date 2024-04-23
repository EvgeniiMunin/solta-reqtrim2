import numpy as np
from typing import Tuple, Generator, List, Dict
from sklearn.metrics import roc_auc_score, r2_score, confusion_matrix

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def time_based_split(
        df: pd.DataFrame,
        n_splits: int,
        hours_train: int,
        hours_val: int
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    try:
        df["ts"] = pd.to_datetime(df["req_ts"])
    except Exception as e:
        logger.error("Failed to convert 'req_ts' to datetime: %s", str(e))
        raise

    try:
        total_hours = hours_train + hours_val
        max_start = max(df["ts"]) - pd.Timedelta(hours=total_hours)
        start_times = pd.date_range(start=min(df["ts"]), end=max_start, periods=n_splits)

        for start_time in start_times:
            train_start = start_time
            train_end = train_start + pd.Timedelta(hours=hours_train)
            val_end = train_end + pd.Timedelta(hours=hours_val)

            train_indices = df[(df["ts"] >= train_start) & (df["ts"] < train_end)].index
            val_indices = df[(df["ts"] >= train_end) & (df["ts"] < val_end)].index

            yield train_indices, val_indices

    except Exception as e:
        logger.error("Failed to split data: %s", str(e))
        raise


def validate_model(
    yval: np.ndarray, ypred: np.ndarray
) -> Tuple[float, float, float, float, float]:
    try:
        auc = roc_auc_score(yval, ypred)
        r2 = r2_score(yval, ypred)
        ypred_mean = ypred.mean()
        ymean = yval.mean()
        calibration = abs(ypred_mean / ymean - 1)

        logger.info(f"AUC: {auc:.4f}, R2: {r2:.4f}, Calibration: {calibration:.4f}")
        logger.info(f"ypred.mean(): {ypred_mean:.4f}, yval.mean(): {ymean:.4f}")

        return auc, r2, ypred_mean, ymean, calibration
    except Exception as e:
        logger.error("Failed to validate model: %s", str(e))
        raise


def validate_filtering(probas: pd.DataFrame, threshold: float):
    probas['preds'] = (probas['ypred'] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(probas['yval'], probas['preds']).ravel()

    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    filter_rate = (tn + fn) / (tp + fn + tn + fp)

    logger.info(f"Filtering metrics - TNR: {tnr:.4f}, TPR: {tpr:.4f}, Filter Rate: {filter_rate:.4f}")