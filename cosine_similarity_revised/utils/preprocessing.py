import re
import unicodedata
from typing import List, Dict


def case_fold_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()


def remove_punctuation(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    normalized = re.sub(r"[-_]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_text(text: str) -> List[str]:
    cleaned = remove_punctuation(text)
    if not cleaned:
        return []
    return cleaned.split()


def preprocess_text(text: str) -> Dict[str, str | List[str]]:
    folded = case_fold_text(text)
    cleaned = remove_punctuation(folded)
    tokens = tokenize_text(cleaned)
    return {
        "original": text,
        "case_folded": folded,
        "without_punctuation": cleaned,
        "tokens": tokens,
    }


def preprocess_texts(texts: List[str]) -> List[Dict[str, str | List[str]]]:
    return [preprocess_text(text) for text in texts]
