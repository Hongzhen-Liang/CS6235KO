from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import pandas as pd
sys.path.append("../Models")
from utils import transformData,chunksize

saveDirName = sys.argv[1]
datasetDir = sys.argv[2]
resultDir = sys.argv[3]
if torch.backends.mps.is_available(): device = "mps"
else:device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
device = torch.device(device)
model = BertForSequenceClassification.from_pretrained(saveDirName).to(device)
tokenizer = BertTokenizer.from_pretrained(saveDirName)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for chunk in pd.read_csv(datasetDir, chunksize=chunksize):
        chunk = chunk[['text', 'label']]
        test_loader = transformData(chunk,tokenizer)
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

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
    print(f"%f,%f,%f,%f\n"%(accuracy,precision,recall,f1))

