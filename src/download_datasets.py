from datasets import load_dataset

LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}
def load_xnli():
    return load_dataset(
        "facebook/xnli",
        "ru"
    )
    
def prepare_split(data):
    data = data.to_pandas()
    data = data[
        [ "premise", "hypothesis", "label"]
    ].copy()
    data["label"] = data["label"].map(LABELS)
    return data

def get_xnli_data():
    data = load_xnli()

    train = prepare_split(data["train"])
    val = prepare_split(data["validation"])
    test = prepare_split(data["test"])

    return train,val,test