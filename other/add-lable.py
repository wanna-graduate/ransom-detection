import pandas as pd

df = pd.read_csv('C:/Users/Xi/perfdata-merged/malicious_sample_normalized.csv')

df['label'] = 1

df.to_csv('C:/Users/Xi/perfdata-merged/malicious_sample_normalized-.csv', index=False)

print("save 'labeled_data.csv'")
