from transformers import BertTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
epochs = 3
chunksize = 1000
batch_size = 100
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def parseData(dirName):
    df = pd.read_csv(dirName)[['text', 'label']]
    return df.to_dict(orient='records')
    


def transformData(data, tokenizer):
    # 将数据转化为 Hugging Face Dataset 格式
    # 将数据集准备为 PyTorch 可用的格式
    tokenized_datasets = Dataset.from_pandas(data).map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])   
    return DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
    

# 定义一个分词函数
def tokenize_function(examples, tokenizer, max_length=128):
    # 这里指定了最大长度和padding的方式，同时开启截断功能
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)