import torch

from src.evidence import check_answer
from src.transformer_test import load_model


MODEL_PATH = "models/rubert_base_nli"


def print_items(title, items):
    print(f"\n{title}:")

    if len(items) == 0:
        print("Нет")
        return

    for item in items:
        print("\nsource:", item["source"])
        print("retrieval_score:", item["retrieval_score"])
        print("nli_label:", item["nli_label"])
        print("nli_confidence:", item["nli_confidence"])
        print("text:", item["text"])


def main():
    documents = [
        {
            "source": "document_1",
            "text": (
                "Компания была основана в 1997 году. "
                "Главный офис компании находится в Москве. "
                "Компания занимается интернет-технологиями."
            )
        },
        {
            "source": "document_2",
            "text": (
                "Некоторые источники утверждают, что компания появилась в 2001 году. "
                "В компании работает несколько тысяч сотрудников."
            )
        },
        {
            "source": "document_3",
            "text": (
                "Основателем организации был российский предприниматель. "
                "Первый продукт был выпущен спустя несколько лет."
            )
        }
    ]

    question = "Когда была основана компания?"

    answer = "Компания была основана в 1997 году."

    model, tokenizer = load_model(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:")
    print(device)

    model.to(device)

    result = check_answer(
        model,
        tokenizer,
        question,
        answer,
        documents,
        top_k=5,
        min_retrieval_score=0.05
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nDecision:")
    print(result["decision"])

    print("\nConfidence:")
    print(result["confidence"])

    print_items("Evidence", result["evidence"])
    print_items("Contradictions", result["contradictions"])
    print_items("Neutral", result["neutral"])


main()