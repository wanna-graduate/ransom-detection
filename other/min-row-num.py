import os
import pandas as pd

def find_min_rows(folder_path):
    min_rows = float('inf')  
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            min_rows = min(min_rows, len(df))
    return min_rows if min_rows != float('inf') else 0  

folder_path = 'C:/Users/Xi/PycharmProjects/blk-perf/perfdata/csv/act-zip'  
min_rows = find_min_rows(folder_path)
print("CSV rows:", min_rows)
