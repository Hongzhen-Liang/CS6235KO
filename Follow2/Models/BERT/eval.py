from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import pandas as pd
sys.path.append("../Models")
from utils import parseData,transformData,chunksize

saveDirName = sys.argv[1]
datasetDir = sys.argv[2]
resultDir = sys.argv[3]
model = BertForSequenceClassification.from_pretrained(saveDirName)
tokenizer = BertTokenizer.from_pretrained(saveDirName)

# test_data = parseData(datasetDir) # [ {"text": "I love programming", "label": 1} ]


# test_loader = transformData(test_data,tokenizer)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for chunk in pd.read_csv(datasetDir, chunksize=chunksize):
        chunk = chunk[['text', 'label']]
        test_loader = transformData(chunk,tokenizer)
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')


with open(resultDir,'w') as f:
    f.write("Accuracy,Precision,Recall,F1 Score\n")
    f.write(f"%f,%f,%f,%f\n"%(accuracy,precision,recall,f1))
    print("Accuracy,Precision,Recall,F1 Score\n")
    print("Accuracy,Precision,Recall,F1 Score\n")

