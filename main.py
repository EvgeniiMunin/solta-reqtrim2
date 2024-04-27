import time
import pandas as pd
import logging
from typing import List, Tuple, Dict
#from tqdm import tqdm
import catboost as cb

from src.data.make_dataset import load_data
from src.features.build_features import hash_feats
from src.models.serialize_model import serialize_predictor, serialize_onnx
from src.models.train_val_model import time_based_split, validate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def train_and_validate(
        df: pd.DataFrame,
        feats: List[str],
        config: Dict[str, int]
) -> Dict[str, list]:
    cv = time_based_split(df, **config)
    val_metrics = {'auc': [], 'r2': [], 'ypred_mean': [], 'ymean': [], 'calibration': []}

    for fold, (train_ids, valid_ids) in enumerate(cv):
        try:
            start_time = time.time()

            train_df, val_df = df.loc[train_ids], df.loc[valid_ids]

            print(f"Fold {fold}")
            print(train_df["ts"].min(), train_df["ts"].max(), train_df.shape)
            print(val_df["ts"].min(), val_df["ts"].max(), val_df.shape)

            model = cb.CatBoostClassifier(
                iterations=1000, learning_rate=0.1, depth=6, l2_leaf_reg=3,
                eval_metric='AUC', random_seed=42, task_type="GPU", verbose=True
            )

            x_train, x_val, y_train, y_val = hash_feats(train_df, val_df, feats)
            
            #model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

            #y_pred = model.predict_proba(x_val)[:, 1]
            #metrics = validate_model(y_val.values, y_pred)

            #for key, value in zip(val_metrics.keys(), metrics):
            #    val_metrics[key].append(value)

            #serialize_predictor(model, f"{config['path_data']}catboost_6h_v1_fold{fold}.pkl")
            #serialize_onnx(model, f"{config['path_data']}catboost_6h_v1_fold{fold}.onnx")

            end_time = time.time()
            duration = end_time - start_time
            print(f"duration: {duration:.2f} seconds \n")

        except Exception as e:
            logger.error(f"Error during training fold {fold}: {str(e)}")

    return val_metrics


def validation_metrics(metrics: Dict[str, list]):
    iterations = list(range(len(metrics["auc"])))
    metrics_df = pd.DataFrame({
        "iteration": iterations,
        "auc": metrics["auc"],
        "r2": metrics["r2"],
        "ypred_mean": metrics["ypred_mean"],
        "ymean": metrics["ymean"],
        "calibration": metrics["calibration"]
    })
    metrics_df.head(50)



if __name__ == "__main__":
    start_time = time.time()
    df = load_data('data/log_13h_rand10.csv')
    logger.info("DataFrame loaded and validated successfully.")
    end_time = time.time()
    duration = end_time - start_time
    print(f"read csv duration: {duration:.2f} seconds")

    feats = [
        "ssp_id", "dsp_id", "creative_type", "dsp_deal_id", "floor",
        "imp", "str", "vtr", "ctr", "vctr",
        "site_id", "domain_id", "tz_offset",
        "api", "browser_version", "region_id", "device_id", "os_id",
        "browser_id"
    ]

    n_splits = 5
    hours_train = 6
    hours_val = 1

    config = {
        'n_splits': n_splits,
        'hours_train': hours_train,
        'hours_val': hours_val,
    }

    metrics = train_and_validate(df, feats,config)
    #validation_metrics(metrics)
