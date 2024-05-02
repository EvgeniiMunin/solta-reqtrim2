import pyarrow.csv as pv
import pyarrow as pa
import pandas as pd
import time

def read_csv():
    start_time = time.time()

    # Create a CSV read options object with multithreading enabled
    read_options = pv.ReadOptions()
    parse_options = pv.ParseOptions()
    convert_options = pv.ConvertOptions()

    # Use the CSV reader from pyarrow with the options specified
    path = 'data/log_13h_rand10.csv'
    table = pv.read_csv(
        path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options
    )
    
    # convert pyarrow to pandas df
    df = table.to_pandas()

    # End the timer
    end_time = time.time()
    
    # Calculate the duration
    duration = end_time - start_time
    print(f"read csv duration: {duration:.2f} seconds")
    print("df.shape: ", df.shape)


def read_multiple_csv():
    paths = [
        'data/rand10/log_2024-04-24 00.csv',
        'data/rand10/log_2024-04-24 01.csv',
        'data/rand10/log_2024-04-24 02.csv',
        'data/rand10/log_2024-04-24 03.csv',
        'data/rand10/log_2024-04-24 04.csv',
        'data/rand10/log_2024-04-24 05.csv',
        'data/rand10/log_2024-04-24 06.csv',
        'data/rand10/log_2024-04-24 07.csv',
        'data/rand10/log_2024-04-24 08.csv',
        'data/rand10/log_2024-04-24 09.csv',
        'data/rand10/log_2024-04-24 10.csv',
        'data/rand10/log_2024-04-24 11.csv',
        'data/rand10/log_2024-04-24 12.csv',
        'data/rand10/log_2024-04-24 13.csv',
        'data/rand10/log_2024-04-24 14.csv',
        'data/rand10/log_2024-04-24 15.csv',
        'data/rand10/log_2024-04-24 16.csv',
        'data/rand10/log_2024-04-24 17.csv',
        'data/rand10/log_2024-04-24 18.csv',
        'data/rand10/log_2024-04-24 19.csv',
        'data/rand10/log_2024-04-24 20.csv',
        'data/rand10/log_2024-04-24 21.csv',
        'data/rand10/log_2024-04-24 22.csv',
        'data/rand10/log_2024-04-24 23.csv',
    ]

    dfs = []

    for path in paths:
        start_time = time.time()
        table = pv.read_csv(path)
        df = table.to_pandas()

        # End the timer
        end_time = time.time()
        duration = end_time - start_time
        print(f"read csv duration: {duration:.2f} seconds")
        print(f"{path} df.shape: ", df.shape, "\n")

        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print("df.shape: ", df.shape)




if __name__ == '__main__':
    read_csv()