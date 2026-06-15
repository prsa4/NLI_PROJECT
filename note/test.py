import pandas as pd
import sys
import torch


sys.path.append("../src")


from transformer_test import (
    evaluate,
    get_data_collator,
    get_data_loader,
    load_model,
    predict_,
    prepare_data
)


MODEL_PATH = "../models/rubert_tiny_nli"


test_data = pd.read_csv(
    "../data/filter/test.csv"
)


print("Test shape:")
print(test_data.shape)

print("\nКлассы test:")
print(
    test_data["label"].value_counts()
)


model, tokenizer = load_model(
    MODEL_PATH
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)


token_test_data = prepare_data(
    test_data,
    tokenizer
)


data_collator = get_data_collator(
    tokenizer
)


test_loader = get_data_loader(
    token_test_data,
    data_collator,
    batch_size=32,
    shuffle=False
)


test_metrics = evaluate(
    model,
    test_loader,
    device
)


print("\nTest loss:")
print(test_metrics["loss"])

print("\nTest accuracy:")
print(test_metrics["accuracy"])

print("\nTest macro F1:")
print(test_metrics["f1_macro"])

print("\nTest contradiction F1:")
print(test_metrics["contradiction_f1"])

print("\nClassification report:")
print(test_metrics["class_report"])

print("\nПорядок классов:")
print([
    "entailment",
    "neutral",
    "contradiction"
])

print("\nConfusion matrix:")
print(test_metrics["confusion_matrix"])


result = predict_(
    model,
    tokenizer,
    premise="Компания была основана в 1997 году.",
    hypothesis="Компания была основана в 2005 году."
)


print("\nПример предсказания:")
print(result)

