import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL_PATH = "../models/rubert_base_nli"
LABEL_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABELS = ["entailment", "neutral", "contradiction"]

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, id2label=ID_LABEL, label2id=LABEL_ID)

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer

def convert_data(data):
    return Dataset.from_pandas(data, preserve_index=False)

def tokenize_data(data, tokenizer):
    token_data = tokenizer(data["premise"], data["hypothesis"], truncation=True, max_length=128)
    token_data["labels"] = [LABEL_ID[label] for label in data["label"]]
    return token_data

def prepare_data(data, tokenizer):
    data = convert_data(data)
    token_data = data.map(lambda batch: tokenize_data(batch, tokenizer), batched=True, remove_columns=data.column_names)
    return token_data

def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def get_data_loader(data, collator, batch_size, shuffle):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

def get_optimizer(model, lr):
    return AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def get_scheduler(optimizer, train_loader, epochs, ac_steps):
    steps_epoch = math.ceil(len(train_loader) / ac_steps)
    steps = steps_epoch * epochs
    warmup_steps = int(steps * 0.1)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler, acc_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for num_batch, batch in enumerate(train_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(**batch)
            loss_ = out.loss
            loss = loss_ / acc_steps

        scaler.scale(loss).backward()
        update = (num_batch % acc_steps == 0 or num_batch == len(train_loader))

        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_.item()
        if num_batch % 100 == 0:
            print(f"Batch {num_batch}/{len(train_loader)}, loss: {loss_.item():.4f}")

    average_loss = total_loss / len(train_loader)
    return average_loss

def evaluate(model, validation_loader, device):
    model.eval()
    total_loss = 0
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch in validation_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(**batch)

            loss = outputs.loss
            pred = torch.argmax(outputs.logits, dim=-1)
            total_loss += loss.item()
            all_label.extend(batch["labels"].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    average_loss = total_loss / len(validation_loader)
    acc = accuracy_score(all_label, all_pred)
    f1_macro = f1_score(all_label, all_pred, average="macro", zero_division=0)
    f1_conf = f1_score(all_label, all_pred, labels=[LABEL_ID["contradiction"]], average="macro", zero_division=0)
    report = classification_report(all_label, all_pred, labels=[0, 1, 2], target_names=LABELS, digits=4, zero_division=0)
    matrix = confusion_matrix(all_label, all_pred, labels=[0, 1, 2])

    return {"loss": average_loss, "accuracy": acc, "f1_macro": f1_macro, "contradiction_f1": f1_conf, "class_report": report, "confusion_matrix": matrix}


def train_model(model, tokenizer, train_data, val_data, epochs, train_batch_size, val_batch_size=32, ac_steps=2, lr=2e-5, path="../models/rubert_base_nli"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2), "GB")

    model.to(device)
    data_collator = get_data_collator(tokenizer)
    train_loader = get_data_loader(train_data, data_collator, train_batch_size, True)
    val_loader = get_data_loader(val_data, data_collator, val_batch_size, False)
    optimizer = get_optimizer(model, lr)
    scheduler = get_scheduler(optimizer, train_loader, epochs, ac_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_f1 = 0

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, ac_steps)
        val_metrics = evaluate(model, val_loader, device)

        print("Train loss:", train_loss)
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

def predict_(model, tokenizer, premise, hypothesis):
    device = next(model.parameters()).device

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    pred = torch.argmax(probabilities).item()

    return {
        "label": ID_LABEL[pred],
        "confidence": float(probabilities[pred]),
        "probabilities": {
            ID_LABEL[class_id]: float(probabilities[class_id])
            for class_id in ID_LABEL
        }
    }