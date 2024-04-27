import pandas as pd
from clickhouse_connect import get_client

def fetch_data_from_clickhouse(start_time: str, end_time: str) -> pd.DataFrame:
    client = get_client(
        host='10.12.0.1',
        port=8123,
        username='stat',
        password='ePDhoXA3Sow1mWRc',
        database='kimberlite_dev'
    )

    query = f"""
        select *
        from ml_flat_log
        where 
            req_ts between '2024-04-23 17:00:00' and '2024-04-24 00:00:00'
            AND rand() % 10 = 1
    """

    result = client.query(query)

    df = pd.DataFrame()

    return df


if __name__ == "__main__":
    start_time = '2024-04-23 17:00:00'
    end_time = '2024-04-24 00:00:00'

    df = fetch_data_from_clickhouse(start_time, end_time)
    print(df.shape)