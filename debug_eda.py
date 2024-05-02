import pandas as pd

"""
TODO
- filter out outliers (ssp, dsp) with bid_rate = 0, 1
- run script on 1h of validation log log_2024-04-25 06
- run EDA on Colab
    - count nb of outliers (ssp, dsp)
    - distribution of reqs, bids, bid_rate per group (ssp, dsp)

Fold 0
2024-04-25 00:14:13 2024-04-25 06:00:00 (5453949, 30)
2024-04-25 06:00:00 2024-04-25 07:00:00 (2807193, 30)

"""

if __name__ == "__main__":
    df = pd.read_csv("data/log_2024-04-25 06.csv")

    biddf = df.groupby(['ssp_id', 'dsp_id']).agg({
        "req_id": "count",
        "bid_state": lambda x: (x.isin(["ok", "ok-proxy"]).astype(int).sum())
    })
    biddf.columns = ['count', 'bids']
    biddf['bid_rate'] = round(biddf['bids'] / biddf['count'], 2)
    biddf.reset_index(inplace=True)
    print(biddf)
    biddf.to_csv("data/ssp_dsp_bids_06.csv", index=False)

    thrdf = pd.read_csv("data/ssp_dsp_thresholds.csv")
    merged_df = pd.merge(thrdf, biddf, on=['ssp_id', 'dsp_id'], how='inner')
    merged_df.to_csv("data/ssp_dsp_bids_thresholds_06.csv", index=False)
