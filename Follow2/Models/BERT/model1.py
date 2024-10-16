from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
import pandas as pd
import sys
sys.path.append("../Models")
from utils import parseData,transformData,tokenizer,epochs,chunksize

saveDirName = sys.argv[1]
datasetDir = sys.argv[2]
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# data = parseData(datasetDir) # [ {"text": "I love programming", "label": 1} ]

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(epochs): 
    for chunk in pd.read_csv(datasetDir, chunksize=chunksize):
        chunk = chunk[['text', 'label']]
        train_loader = transformData(chunk,tokenizer)
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
model.save_pretrained(saveDirName)
tokenizer.save_pretrained(saveDirName)
print("Training complete!")
