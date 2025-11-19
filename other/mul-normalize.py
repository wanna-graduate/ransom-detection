import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

folder_path = Path('D:/ranote/perf/ranote-data/csv') 

csv_files = list(folder_path.glob('*.csv')) 

all_dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    all_dfs.append(df)

combined_df = pd.concat(all_dfs)

scaler = MinMaxScaler()
scaler.fit(combined_df)


# 加载你要归一化的指定文件
file_path = 'D:/3/7z-train.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 对指定文件的数据进行归一化
normalized_data = scaler.transform(df)

# 将归一化后的数据转换为DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

# 将归一化后的数据保存为CSV文件
output_file_path = 'D:/ranote/perf/ranote-data'  # 设置输出文件的路径
normalized_df.to_csv(output_file_path, index=False)

print(f"归一化后的数据已保存到 {output_file_path}")
