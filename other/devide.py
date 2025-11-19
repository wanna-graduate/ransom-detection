import pandas as pd

def split_csv(file_path, output_file_80, output_file_20):

    df = pd.read_csv(file_path)
    chunk_size = 270
    head_size = int(chunk_size * 0.8)
    tail_size = chunk_size - head_size
    head_df = pd.DataFrame()
    tail_df = pd.DataFrame()

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        head_part = chunk.head(head_size)
        tail_part = chunk.tail(tail_size)
        head_df = pd.concat([head_df, head_part], ignore_index=True)
        tail_df = pd.concat([tail_df, tail_part], ignore_index=True)

    head_df.to_csv(output_file_80, index=False)
    tail_df.to_csv(output_file_20, index=False)
    print(f"80% {output_file_80}")
    print(f"20% {output_file_20}")

split_csv('C:/Users/Xi/PycharmProjects/blk-perf/blk/csv-/benign-merged.csv', 'C:/Users/Xi/PycharmProjects/blk-perf/blk/benign-80.csv', 'C:/Users/Xi/PycharmProjects/blk-perf/blk/benign-20.csv')


