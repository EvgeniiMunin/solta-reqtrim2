import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import logging
import joblib

from typing import Tuple

from src.features.build_features import apply_feature_hashing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_tpr_threshold(
    yval: np.ndarray, ypred: np.ndarray, target_tnr: float
) -> Tuple[float, float, float]:
    probas = pd.DataFrame({
        "yval": yval,
        "ypred": ypred
    })

    negative_probas = probas[probas['yval'] == 0]
    threshold = negative_probas['ypred'].quantile(target_tnr)

    positive_preds = probas["ypred"] >= threshold
    tp = (probas["yval"] == 1) & positive_preds
    fp = (probas["yval"] == 0) & positive_preds
    tn = (probas["yval"] == 0) & ~positive_preds
    fn = (probas["yval"] == 1) & ~positive_preds

    negatives_count = tp.sum() + fn.sum()
    total_count = tp.sum() + fn.sum() + tn.sum() + fp.sum()
    tpr = tp.sum() / negatives_count if negatives_count > 0 else 0
    filter_rate = (tn.sum() + fn.sum()) / total_count if total_count > 0 else 0

    return threshold, tpr, filter_rate

def compute_thresholds_by_group(df, ssp_id, dsp_id):
    tnr_targets = np.arange(0.05, 1, 0.05)
    results = []
    for tnr_target in tnr_targets:
        threshold, tpr, filter_rate = compute_tpr_threshold(
            df["yval"].values, df["ypred"].values, tnr_target
        )

        print(f"----tnr_target: {round(tnr_target, 4)}, thr: {round(threshold, 4)}, TPR@TNR: {round(tpr, 4)}, filter_rate: {round(filter_rate, 4)}")

        results.append({
            "ssp_id": ssp_id,
            "dsp_id": dsp_id,
            "TNR_percentile": round(tnr_target * 100, 4),
            "threshold": round(threshold, 4),
            "TPR@TNR": round(tpr, 4),
            "filter_rate": round(filter_rate, 4)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    feats = [
        "ssp_id", "dsp_id", "creative_type", "dsp_deal_id", "floor",
        "imp", "str", "vtr", "ctr", "vctr",
        "site_id", "domain_id", "tz_offset",
        "api", "browser_version", "region_id", "device_id", "os_id",
        "browser_id"
    ]

    df = pd.read_csv("data/log_2024-04-25 06.csv")
    xval = apply_feature_hashing(df[feats])
    yval = df["bid_state"].isin(["ok", "ok-proxy"]).astype(int)

    model = joblib.load(f"models/catboost_6h_v1_fold0.pkl")
    print("df.shape: ", df.shape, "\n", df.info())
    print("model: ", model)

    ypred = model.predict_proba(xval)[:, 1]
    df["ypred"] = ypred
    df["yval"] = yval
    print("df with preds: \n", df[["ssp_id", "dsp_id", "bid_state", "ypred", "yval"]])

    results = []
    for (ssp_id, dsp_id), group in df.groupby(["ssp_id", "dsp_id"]):
        print(f"ssp_id: {ssp_id}, dsp_id: {dsp_id}, group: {group.shape}")
        result_df = compute_thresholds_by_group(group, ssp_id, dsp_id)
        results.append(result_df)

    final_results = pd.concat(results).reset_index(drop=True)

    print("final_results: \n",final_results)
    final_results.to_csv("data/ssp_dsp_thresholds_06.csv", index=False)