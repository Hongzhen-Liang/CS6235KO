from transformers import BertTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd

chunksize = 1000
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_config = {
    "data_param": {
        "dataset": "time_sorted",
        "max_data_size": -1,
        "batch_size": 8,
        "data_root": "/content/drive/My Drive/KO/",
        "train_datapath": "1-2020",
        "val_datapath": "",
        "test_datapath": "2-2020",
        "num_classes": 2,
        "filter_long_text": True
    },
    "model": "bert",
    "tokenizer": "bert-base-uncased",
    "model_param": {
        "vocab_size": 30522,
        "embedding_size": 128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_hidden_groups": 1,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "inner_group_num": 1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1.0e-12,
        "classifier_dropout": 0.1
    },
    "trainer_param": {
        "epochs": 10,
        "val_epochs": 1,
        "loss_func": "cross_entropy",
        "metric": "acc",
        "optimizer": "AdamW",
        "optimizer_param": {
            "lr": 1.0e-5,
            "eps": 1.0e-6,
            "weight_decay": 0.0005
        }
    }
}

def parseData(dirName):
    df = pd.read_csv(dirName)[['text', 'label']]
    return df.to_dict(orient='records')
    


def transformData(data, tokenizer):
    # 将数据转化为 Hugging Face Dataset 格式
    # 将数据集准备为 PyTorch 可用的格式
    tokenized_datasets = Dataset.from_pandas(data).map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])   
    return DataLoader(tokenized_datasets, batch_size=bert_config["data_param"]["batch_size"], shuffle=True)
    

# 定义一个分词函数
def tokenize_function(examples, tokenizer):
    # 这里指定了最大长度和padding的方式，同时开启截断功能
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=tokenizer.model_max_length)