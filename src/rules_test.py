import re


NEGATION_WORDS = {
    "не",
    "нет",
    "никогда",
    "ничего",
    "никто",
    "нигде",
    "нельзя",
    "без"
}


def get_words(text):
    return re.findall(
        r"[а-яёa-z]+",
        text.lower()
    )


def extract_numbers(text):
    numbers = re.findall(
        r"\d+(?:[.,]\d+)?",
        text
    )

    return {
        number.replace(",", ".")
        for number in numbers
    }


def has_negation(text):
    words = set(
        get_words(text)
    )

    return bool(
        words & NEGATION_WORDS
    )


def get_word_overlap(premise, hypothesis):
    premise_words = set(get_words(premise))

    hypothesis_words = set(get_words(hypothesis))

    if not premise_words or not hypothesis_words:
        return 0.0

    common_words = (premise_words& hypothesis_words)

    all_words = (premise_words| hypothesis_words)

    return len(common_words) / len(all_words)


def analyze_pair(premise, hypothesis):
    premise_numbers = extract_numbers(premise)

    hypothesis_numbers = extract_numbers(hypothesis)

    premise_negation = has_negation(premise)

    hypothesis_negation = has_negation(hypothesis)
 
    word_overlap = get_word_overlap(premise,hypothesis )

    number_mismatch = (bool(premise_numbers) and bool(hypothesis_numbers) and premise_numbers != hypothesis_numbers)

    negation_mismatch = (premise_negation != hypothesis_negation)

    return {
        "premise_numbers": premise_numbers,
        "hypothesis_numbers": hypothesis_numbers,
        "number_mismatch": number_mismatch,
        "premise_negation": premise_negation,
        "hypothesis_negation": hypothesis_negation,
        "negation_mismatch": negation_mismatch,
        "word_overlap": word_overlap
    }


def get_error_category(premise, hypothesis):
    analysis = analyze_pair(
        premise,
        hypothesis
    )

    if analysis["number_mismatch"]:
        return "number_mismatch"

    if analysis["negation_mismatch"]:
        return "negation_mismatch"

    if analysis["word_overlap"] < 0.1:
        return "low_word_overlap"

    return "other"