from snorkel.labeling import labeling_function,LFApplier
from snorkel.labeling.model import LabelModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import BertForSequenceClassification,BertTokenizer
import sys
import pandas as pd
import os

datasetDir = sys.argv[1]
resultDir = sys.argv[2]
output_path = sys.argv[3]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@labeling_function()
def model_bert1_lf(x):
    saveDirName = "../../Follow2/Save_Models/bert1"
    model = BertForSequenceClassification.from_pretrained(saveDirName).to(device)
    tokenizer = BertTokenizer.from_pretrained(saveDirName)
    inputs = tokenizer(
        x,
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label
@labeling_function()
def model_bert2_lf(x):
    saveDirName = "../../Follow2/Save_Models/bert2"
    model = BertForSequenceClassification.from_pretrained(saveDirName).to(device)
    tokenizer = BertTokenizer.from_pretrained(saveDirName)
    inputs = tokenizer(
        x,
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label
@labeling_function()
def model_bert3_lf(x):
    saveDirName = "../../Follow2/Save_Models/bert3"
    model = BertForSequenceClassification.from_pretrained(saveDirName).to(device)
    tokenizer = BertTokenizer.from_pretrained(saveDirName)
    inputs = tokenizer(
        x,
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label
@labeling_function()
def model_bert4_lf(x):
    saveDirName = "../../Follow2/Save_Models/bert4"
    model = BertForSequenceClassification.from_pretrained(saveDirName).to(device)
    tokenizer = BertTokenizer.from_pretrained(saveDirName)
    inputs = tokenizer(
        x,
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label

@labeling_function()
def model_bert6_lf(x):
    saveDirName = "../../Follow4/Save_Models/bert6"
    model = BertForSequenceClassification.from_pretrained(saveDirName).to(device)
    tokenizer = BertTokenizer.from_pretrained(saveDirName)
    inputs = tokenizer(
        x,
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label

# Apply the LFs to your dataset
df = pd.read_csv(datasetDir, names=["text", "label"], skiprows=1)
df_train = df["text"]
# print("df_train shape:", df_train.shape)

lfs = [model_bert1_lf, model_bert2_lf, model_bert3_lf, model_bert4_lf, model_bert6_lf]
applier = LFApplier(lfs=lfs)
L_train = applier.apply(df_train)
label_model = LabelModel(cardinality=2)  # Assuming binary labels: FAKE, REAL
label_model.fit(L_train, n_epochs=100, lr=0.01)
probs_train = label_model.predict_proba(L_train)
# print(probs_train)
all_labels = df["label"].values
all_preds = probs_train.argmax(axis=-1)
# print(all_labels)
# print(all_preds)
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
os.makedirs(os.path.dirname(resultDir), exist_ok=True)
with open(resultDir,'w') as f:
    f.write("Accuracy,Precision,Recall,F1 Score\n")
    f.write(f"%f,%f,%f,%f\n"%(accuracy,precision,recall,f1))
    print("Accuracy,Precision,Recall,F1 Score\n")
    print(f"%f,%f,%f,%f\n"%(accuracy,precision,recall,f1))

df['label'] = all_preds
df.to_csv(output_path, index=False)

