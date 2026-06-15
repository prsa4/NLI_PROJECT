from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader

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

def get_data_loader(data,collator,batch_size,shuffle):
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle, collate_fn=collator)

def get_optimizer(model, lr):
    return AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def get_scheduler(optimizer,train_loader,epochs):
    steps = len(train_loader)*epochs
    warmup_steps = int(steps*0.1)
    return get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=steps)

def train_epoch(model, train_loader, optimizer,scheduler,device="cuda"):
    model.train()
    total_loss=0
    
    for num_batch, batch in enumerate(train_loader, start=1):
        batch = {
            k: v.to(device)
            for k,v in batch.items()
        }
        optimizer.zero_grad()
        out = model(**batch)
        loss =out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if num_batch % 100 == 0:
            print(
                f"Batch {num_batch}/"
                f"{len(train_loader)}, "
                f"loss: {loss.item():.4f}"
            )

        average_loss = (
            total_loss / len(train_loader)
        )

    return average_loss
 
def evaluate(model, validation_loader, device):
    model.eval()

    total_loss = 0

    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch in validation_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            pred = torch.argmax(
                outputs.logits,
                dim=-1
            )

            total_loss += loss.item()

            all_label.extend(
                batch["labels"]
                .cpu()
                .numpy()
            )

            all_pred.extend(
                pred
                .cpu()
                .numpy()
            )

    average_loss = (
        total_loss
        / len(validation_loader)
    )

    acc = accuracy_score(
        all_label,
        all_pred
    )

    f1_macro = f1_score(
        all_label,
        all_pred,
        average="macro",
        zero_division=0
    )

    f1_conf = f1_score(
        all_label,
        all_pred,
        labels=[
            LABEL_ID["contradiction"]
        ],
        average="macro",
        zero_division=0
    )

    return {
        "loss": average_loss,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "contradiction_f1": f1_conf
    }

def train_model(model,tokenizer,train_data, val_data,epochs,train_batch_size, val_batch_size=32,lr=2e-5, path="../models/rubert_tiny_nli"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_collator = get_data_collator(tokenizer)
    train_loader = get_data_loader(train_data,data_collator,train_batch_size,True)
    val_loader = get_data_loader(val_data,data_collator,val_batch_size,True)
    optimizer = get_optimizer(model,lr)
    scheduler = get_scheduler(optimizer,train_loader,epochs)

    best_f1=0
    for epoch in range(epochs):
        print("Epoch:", epoch)
        train_loss = train_epoch(model,train_loader,optimizer,scheduler,device)
        val_metrics=evaluate(model,val_loader,device)
        print("LOSS:", train_loss)
        print("Val loss:", val_metrics["loss"])
        print("Val accuracy:", val_metrics["accuracy"])
        print("Validation macro F1:", val_metrics["f1_macro"])
        print("Validation contradiction F1:", val_metrics["contradiction_f1"])

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            print("Лучшая модель сохранена")

    return model

