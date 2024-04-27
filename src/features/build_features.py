import pandas as pd
import numpy as np
from typing import Dict, Tuple
import pymmh3 as mmh3
import catboost
from sklearn.feature_extraction import FeatureHasher

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_hashing_trick(
        df: pd.DataFrame,
        unique_values: Dict[str, int]
) -> pd.DataFrame:
    result = {}
    for col in df.columns:
        if unique_values[col] <= 0:
            logger.warning(f"Skipping column {col} due to non-positive unique value count")
            continue

        try:
            hashed_values = df[col].apply(lambda x: mmh3.hash(str(x)) % unique_values[col])
            result[col] = hashed_values
        except Exception as e:
            logger.error(f"Error processing column {col}: {str(e)}")
            raise

    return pd.DataFrame(result)

def apply_feature_hashing(df: pd.DataFrame, n_features: int = 2 ** 10) -> pd.DataFrame:
  hasher = FeatureHasher(n_features=n_features, input_type="string")
  hashed_features = hasher.transform(df.astype(str).values)
  hashed_df = pd.DataFrame(hashed_features.toarray())

  hashed_df.columns = [f'feature_{i}' for i in range(hashed_df.shape[1])]
  return hashed_df


def hash_feats(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feats: list
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    #unique_values = train_df[feats].nunique()

    #train_df_hashed = compute_hashing_trick(train_df[feats], unique_values)
    #val_df_hashed = compute_hashing_trick(val_df[feats], unique_values)

    train_df_hashed = apply_feature_hashing(train_df[feats])
    val_df_hashed = apply_feature_hashing(val_df[feats])
    feats = train_df_hashed.columns

    train_df_hashed["bid_state"] = train_df.reset_index(drop=True)["bid_state"]
    val_df_hashed["bid_state"] = val_df.reset_index(drop=True)["bid_state"]

    #logger.info("unique_values: \n", train_df[feats].nunique(), "\n")
    #logger.info("unique_values_train_feats: \n", train_df_hashed.nunique(), "\n")

    x_train = train_df_hashed[feats]
    x_val = val_df_hashed[feats]
    y_train = train_df_hashed["bid_state"].isin(["ok", "ok-proxy"]).astype(int)
    y_val = val_df_hashed["bid_state"].isin(["ok", "ok-proxy"]).astype(int)

    logger.info(f"Train bid rate: {y_train.sum()}, {y_train.mean() * 100:.4f}%")
    logger.info(f"Validation bid rate: {y_val.sum()}, {y_val.mean() * 100:.4f}%")

    return x_train, x_val, y_train, y_val
