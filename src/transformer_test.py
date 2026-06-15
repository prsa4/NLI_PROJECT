from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)
import torch

MODEL_NAME = "cointegrated/rubert-tiny2"
LABEL_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

ID_LABEL = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, id2label=ID_LABEL, label2id=LABEL_ID)

def convert_data(data):
    return Dataset.from_pandas(
        data,
        preserve_index=False
    )

def tokenize_data(data, tokenizer):
    token_data = tokenizer(data["premise"], data["hypothesis"],truncation=True ,max_length=128 )
    token_data["labels"] = [LABEL_ID[label] for label in data["label"]]
    return token_data

def prepare_data(data,tokenizer):
    data = convert_data(data)
    token_data = data.map(
        lambda batch: tokenize_data(batch,tokenizer),
        batched=True,
        remove_columns=data.column_names
    )
    return token_data

def get_data_collator(tokenizer):
    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )


def tokenize_test(tokenizer, premise,hypothesis):
     return tokenizer(premise, hypothesis, truncation=True, max_length=128)


def forward_batch(model, batch):
    model.eval()
    with torch.no_grad():
        out = model(**batch)

    proba = torch.softmax(out.logits,dim=-1)
    pred=torch.argmax(proba, dim=-1)

    return out, proba, pred
