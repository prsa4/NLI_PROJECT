import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() != ""
    ]

    return sentences


def prepare_sentences(documents):
    sentences = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_sentences = split_sentences(text)

        for sentence in document_sentences:
            sentences.append({
                "text": sentence,
                "source": source
            })

    return sentences


def find_relevant_sentences(question, documents, top_k=5):
    sentences = prepare_sentences(documents)

    if len(sentences) == 0:
        return []

    texts = [
        sentence["text"]
        for sentence in sentences
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    sentence_vectors = vectorizer.fit_transform(texts)

    question_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        question_vector,
        sentence_vectors
    )[0]

    result = []

    for sentence, score in zip(
        sentences,
        similarities
    ):
        result.append({
            "text": sentence["text"],
            "source": sentence["source"],
            "score": float(score)
        })

    result = sorted(
        result,
        key=lambda item: item["score"],
        reverse=True
    )

    return result[:top_k]