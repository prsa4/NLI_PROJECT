from src.retrieval import find_relevant_sentences
from src.transformer_test import predict_


def check_answer(model, tokenizer, question, answer, documents, top_k=5, min_retrieval_score=0.05):
    relevant_sentences = find_relevant_sentences(question, documents, top_k=top_k)

    evidence = []
    contradictions = []
    neutral = []

    for item in relevant_sentences:
        if item["score"] < min_retrieval_score:
            continue

        result = predict_(
            model,
            tokenizer,
            premise=item["text"],
            hypothesis=answer
        )

        checked_item = {
            "text": item["text"],
            "source": item["source"],
            "retrieval_score": item["score"],
            "nli_label": result["label"],
            "nli_confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }

        if result["label"] == "entailment":
            evidence.append(checked_item)

        elif result["label"] == "contradiction":
            contradictions.append(checked_item)

        else:
            neutral.append(checked_item)

    confidence = calculate_confidence(evidence, contradictions)
    decision = make_decision(evidence, contradictions, confidence)

    return {
        "question": question,
        "answer": answer,
        "decision": decision,
        "confidence": confidence,
        "evidence": evidence,
        "contradictions": contradictions,
        "neutral": neutral
    }


def calculate_confidence(evidence, contradictions):
    if len(evidence) == 0:
        return 0.0

    support_score = sum(
        item["nli_confidence"] * item["retrieval_score"]
        for item in evidence
    )

    contradiction_score = sum(
        item["nli_confidence"] * item["retrieval_score"]
        for item in contradictions
    )

    score = support_score - contradiction_score

    score = max(0.0, score)
    score = min(1.0, score)

    return float(score)


def make_decision(evidence, contradictions, confidence):
    if len(evidence) == 0:
        return "abstain"

    if confidence < 0.3:
        return "abstain"

    return "answer"