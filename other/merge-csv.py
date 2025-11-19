import os
import pandas as pd


input_folder = 'C:/Users/Xi/PycharmProjects/blk-perf/perfdata/csv/mal-/train'  
output_file = '/blk-perf/perfdata/csv/mal-/mal-merged.csv'  


merged_data = pd.DataFrame()

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        print(filename)
        data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

merged_data.to_csv(output_file, index=True)

print(f'save {output_file}')
