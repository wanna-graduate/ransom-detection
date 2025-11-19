import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#from pythonProject.DL.lstm import malicious_data

benign_df = pd.read_csv('C:/Users/Xi/PycharmProjects/blk-perf/perf/benign-merged.csv')
malicious_df = pd.read_csv('C:/Users/Xi/PycharmProjects/blk-perf/perf/mal-merged.csv')
unknown_df = pd.read_csv('C:/Users/Xi/PycharmProjects/blk-perf/perf/unknown-merged.csv')

combined_df = pd.concat([benign_df, malicious_df, unknown_df])

scaler = MinMaxScaler()
scaler.fit(combined_df)

benign_normalized = pd.DataFrame(scaler.transform(benign_df), columns=benign_df.columns)
mal_normalized = pd.DataFrame(scaler.transform(malicious_df), columns=malicious_df.columns)
unknown_normalized = pd.DataFrame(scaler.transform(unknown_df), columns=unknown_df.columns)

scaling_params = pd.DataFrame({
    'Feature': benign_df.columns,
    'Min': scaler.data_min_,
    'Max': scaler.data_max_
})

print(scaling_params)

#benign_normalized.to_csv('C:/Users/Xi/PycharmProjects/blk-perf/perf/benign-nor.csv', index=False)
#mal_normalized.to_csv('C:/Users/Xi/PycharmProjects/blk-perf/perf/mal-nor.csv', index=False)
#unknown_normalized.to_csv('C:/Users/Xi/PycharmProjects/blk-perf/perf/unknown-nor.csv', index=False)
