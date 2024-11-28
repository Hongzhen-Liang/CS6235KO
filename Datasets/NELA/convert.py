import pandas as pd

fileName = '10-2020.csv'
df = pd.read_csv('ori_'+fileName)
df = df[['text', 'label']]
df.to_csv(fileName, index=False)

