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


def time_based_splits2(paths, num_splits=6, train_hours=6, val_hours=1):
    for fold in range(num_splits):
        # Calculate the index for training and validation split
        train_start_idx = fold
        train_end_idx = train_start_idx + train_hours
        val_end_idx = train_end_idx + val_hours

        # Ensure we do not go out of the list's range
        if val_end_idx > len(paths):
            break

        # Select paths for training and validation
        train_files = paths[train_start_idx:train_end_idx]
        val_files = paths[train_end_idx:val_end_idx]

        # Read training and validation data
        train_df = pd.concat([load_data(f) for f in train_files], ignore_index=True)
        val_df = pd.concat([load_data(f) for f in val_files], ignore_index=True)

        train_df["ts"] = pd.to_datetime(train_df["req_ts"])
        val_df["ts"] = pd.to_datetime(val_df["req_ts"])

        print(f"Fold {fold}")
        print(train_df["ts"].min(), train_df["ts"].max(), train_df.shape)
        print(val_df["ts"].min(), val_df["ts"].max(), val_df.shape)

        yield train_df, val_df

if __name__ == "__main__":
    paths = [
        f'data/rand10/log_2024-04-25 {hour:02}.csv' for hour in range(13)
    ]

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

    for train_df, val_df in time_based_splits2(paths):
        start_time = time.time()

        print(f'Train shape: {train_df.shape}, Validation shape: {val_df.shape}')

        train_df = train_df.sample(frac=0.05).reset_index(drop=True)
        val_df = val_df.sample(frac=0.05).reset_index(drop=True)

        x_train, x_val, y_train, y_val = hash_feats(train_df, val_df, feats)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"hash duration: {duration:.2f} seconds")

        break

        model = cb.CatBoostClassifier(
            iterations=400,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,  # Regularization term
            eval_metric='AUC',  # Metric for evaluation during training
            random_seed=42,
            #task_type="GPU",  # Indicate that training should be done on GPU
            verbose=200  # Output the training progress every 200 iterations
        )

        start_time = time.time()

        model.fit(
            xtrain, ytrain,
            eval_set=(xval, yval),
            use_best_model=True
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f"train duration: {duration:.2f} seconds")

        ypred = model.predict_proba(xval)[:, 1]
        auc, r2, ypred_mean, ymean, calibration = validate_model(yval, ypred)
        val_auc.append(auc)
        val_r2.append(r2)
        val_ypred_mean.append(ypred_mean)
        val_ymean.append(ymean)
        val_calibration.append(calibration)

        break

    #metrics = train_and_validate(df, feats, config)
    #validation_metrics(metrics)
