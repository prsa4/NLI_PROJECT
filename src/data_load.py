import pandas as pd

COLUMNS = {
    "premise", "hypothesis", "label"
}

LABELS = {
    "entailment", "contradiction", "neutral"
}

def load(path):
    return pd.read_csv(path)


def clean_data(db):
    data = db.copy()
    data = data.dropna(
        subset=[
            "premise", "hypothesis", "label"
        ]
    )
    data["premise"] = data["premise"].astype(str)
    data["hypothesis"] = data["hypothesis"].astype(str)
    data["label"] = data["label"].astype(str)

    data["premise"] = data["premise"].str.strip()
    data["hypothesis"] = data["hypothesis"].str.strip()
    data["label"] = data["label"].str.strip().str.lower()

    data = data[(data["premise"] != "") & (data["hypothesis"] != "")]
    data = data[data["label"].isin(LABELS)]
    data = data.drop_duplicates()
    return data.reset_index(drop=True)

def get_data(path):
    data = load(path)
    return clean_data(data)




