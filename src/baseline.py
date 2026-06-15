import joblib


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.pipeline import Pipeline




LABELS = [
    "entailment",
    "contradiction",
    "neutral"
]




def combine_text(data):
    return data["premise"] + " [SEP] " + data["hypothesis"]




def create_baseline():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])




def train_baseline(data):
    train_text = combine_text(data)
    train_label = data["label"]


    model = create_baseline()


    model.fit(
        train_text,
        train_label
    )


    return model




def evaluate(model, data):
    text = combine_text(data)
    label = data["label"]


    pred = model.predict(text)


    acc = accuracy_score(
        label,
        pred
    )


    f1_macro = f1_score(
        label,
        pred,
        average="macro",
        zero_division=0
    )


    f1_conf = f1_score(
        label,
        pred,
        labels=["contradiction"],
        average="macro",
        zero_division=0
    )


    report = classification_report(
        label,
        pred,
        digits=4,
        zero_division=0
    )


    matrix = confusion_matrix(
        label,
        pred,
        labels=LABELS
    )


    prediction_distribution = {
        current_label: int(
            (pred == current_label).sum()
        )
        for current_label in LABELS
    }


    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "contradiction_f1": f1_conf,
        "class_report": report,
        "confusion_matrix": matrix,
        "prediction_distribution": prediction_distribution
    }




def predict_(model, premise, hypothesis):
    text = premise + " [SEP] " + hypothesis


    label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]


    classes = model.named_steps["classifier"].classes_


    probability_dict = dict(
        zip(classes, probabilities)
    )


    conf = probability_dict[label]


    return {
        "label": label,
        "confidence": float(conf),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability
            in probability_dict.items()
        }
    }




def save_baseline(model, path):
    joblib.dump(model, path)




def load_baseline(path):
    return joblib.load(path)

