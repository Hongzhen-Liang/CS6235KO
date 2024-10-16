import pandas as pd

fileName = '10-2020.csv'
# 读取 CSV 文件
df = pd.read_csv('ori_'+fileName)

# 选择保留的列：text 和 label
df = df[['text', 'label']]

# 保存处理后的数据到新的 CSV 文件
df.to_csv(fileName, index=False)

