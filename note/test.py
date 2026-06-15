import pandas as pd
import sys

from sklearn.model_selection import (
    train_test_split
)


sys.path.append("../src")


from transformer_test import (
    get_model,
    get_tokenizer,
    prepare_data,
    train_model
)


train_data = pd.read_csv(
    "../data/filter/train.csv"
)

validation_data = pd.read_csv(
    "../data/filter/val.csv"
)


print("Полный train:")
print(train_data.shape)

print("\nValidation:")
print(validation_data.shape)


small_train_data, _ = train_test_split(
    train_data,
    train_size=30000,
    random_state=42,
    stratify=train_data["label"]
)


small_train_data = (
    small_train_data
    .reset_index(drop=True)
)


print("\nTrain для первого запуска:")
print(small_train_data.shape)

print("\nКлассы train:")
print(
    small_train_data[
        "label"
    ].value_counts()
)

print("\nКлассы validation:")
print(
    validation_data[
        "label"
    ].value_counts()
)


tokenizer = get_tokenizer()

model = get_model()


token_train_data = prepare_data(
    small_train_data,
    tokenizer
)

token_validation_data = prepare_data(
    validation_data,
    tokenizer
)


print("\nТокенизированный train:")
print(token_train_data)

print("\nТокенизированный validation:")
print(token_validation_data)


model = train_model(
    model,
    tokenizer,
    token_train_data,
    token_validation_data,
    epochs=2,
    train_batch_size=16,
    val_batch_size=32,
    lr=2e-5,
    path="../models/rubert_tiny_nli"
)

