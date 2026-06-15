import pandas as pd
import sys

sys.path.append("../src")

from transformer_test import (
    ID_LABEL,
    forward_batch,
    get_data_collator,
    get_model,
    get_tokenizer,
    prepare_data
)

train_data = pd.read_csv("../data/filter/train.csv")
tokenizer=get_tokenizer()
model=get_model()

token_train_data=prepare_data(train_data.head(),tokenizer)
print("token_train_data\n", token_train_data)
data_collator=get_data_collator(tokenizer)
example = [token_train_data[i] for i in range(5)]
batch = data_collator(example)

for key, value in batch.items():
    print(
        key,
        value.shape
    )

outputs,probabilities,pred= forward_batch(model,batch)

print("\nLoss:")
print(outputs.loss.item())

print("\nLogits:")
print(outputs.logits)

print("\nВероятности:")
print(probabilities)

print("\nЧисловые предсказания:")
print(pred)

print("\nРезультаты по примерам:")

for i in range(len(pred)):
    true_id = batch["labels"][i].item()
    pred_id = pred[i].item()

    print(f"\nПример {i + 1}")
    print(
        "Правильный класс:",
        ID_LABEL[true_id]
    )
    print(
        "Предсказанный класс:",
        ID_LABEL[pred_id]
    )
    print(
        "Вероятности:",
        {
            ID_LABEL[class_id]: float(
                probabilities[i][class_id]
            )
            for class_id in ID_LABEL
        }
    )
