import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig
import pandas as pd
import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+"/../")
from utils import bert_config,transformData,tokenizer,chunksize

saveDirName = sys.argv[1]
datasetDir = sys.argv[2]
config = BertConfig(
    vocab_size=bert_config["model_param"]["vocab_size"],
    hidden_size=bert_config["model_param"]["hidden_size"],
    num_hidden_layers=bert_config["model_param"]["num_hidden_layers"],
    num_attention_heads=bert_config["model_param"]["num_attention_heads"],
    intermediate_size=bert_config["model_param"]["intermediate_size"],
    hidden_act=bert_config["model_param"]["hidden_act"],
    hidden_dropout_prob=bert_config["model_param"]["hidden_dropout_prob"],
    attention_probs_dropout_prob=bert_config["model_param"]["attention_probs_dropout_prob"],
    max_position_embeddings=bert_config["model_param"]["max_position_embeddings"],
    type_vocab_size=bert_config["model_param"]["type_vocab_size"],
    initializer_range=bert_config["model_param"]["initializer_range"],
    layer_norm_eps=bert_config["model_param"]["layer_norm_eps"],
    num_labels=bert_config["data_param"]["num_classes"]
)

if torch.backends.mps.is_available(): device = "mps"
else:device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
device = torch.device(device)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',config=config).to(device)

optimizer = AdamW(
    model.parameters(), 
    lr=bert_config["trainer_param"]["optimizer_param"]["lr"],  
    eps=bert_config["trainer_param"]["optimizer_param"]["eps"],  
    weight_decay=bert_config["trainer_param"]["optimizer_param"]["weight_decay"] 
)
model.train()
criterion = torch.nn.BCEWithLogitsLoss()  
for epoch in range(bert_config["trainer_param"]["epochs"]): 
    for chunk in pd.read_csv(datasetDir, chunksize=chunksize):
        chunk = chunk[['text', 'label']]
        train_loader = transformData(chunk,tokenizer)
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            labels_one_hot = F.one_hot(labels, num_classes=bert_config["data_param"]["num_classes"]).float()

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits,labels_one_hot)

            loss.backward()

            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
model.save_pretrained(saveDirName)
tokenizer.save_pretrained(saveDirName)
print("Training complete!")
