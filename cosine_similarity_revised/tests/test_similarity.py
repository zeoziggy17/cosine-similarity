import numpy as np
import pandas as pd

from utils.preprocessing import case_fold_text, remove_punctuation, tokenize_text
from utils.similarity import cosine_similarity, create_binary_matrix, compute_similarity_pairs, build_similarity_matrix


def test_case_folding_and_punctuation_removal():
    text = "Ayu dan Budi! Apakah data-mining itu menarik?"
    cleaned = remove_punctuation(case_fold_text(text))
    assert "data mining" in cleaned.lower()
    assert "?" not in cleaned


def test_tokenization_keeps_words_separated():
    text = "Data-mining dan data mining"
    tokens = tokenize_text(text)
    assert tokens == ["data", "mining", "dan", "data", "mining"]


def test_cosine_similarity_identical_texts():
    vec_a = np.array([1, 0, 1])
    vec_b = np.array([1, 0, 1])
    assert np.isclose(cosine_similarity(vec_a, vec_b), 1.0)


def test_cosine_similarity_different_texts():
    vec_a = np.array([1, 1, 0])
    vec_b = np.array([0, 0, 1])
    assert np.isclose(cosine_similarity(vec_a, vec_b), 0.0)


def test_binary_matrix_and_pairs():
    preprocessed = [
        {"tokens": ["saya", "suka", "makan", "pisang"]},
        {"tokens": ["saya", "suka", "minum", "teh"]},
        {"tokens": ["saya", "makan", "pisang"]},
    ]
    vocab = sorted({token for item in preprocessed for token in item["tokens"]})
    binary_df = create_binary_matrix(preprocessed, vocab)
    assert binary_df.shape == (3, 6)
    assert binary_df.iloc[0].sum() == 4
    pairs = compute_similarity_pairs(binary_df)
    assert len(pairs) == 3
    assert pairs["Cosine Similarity"].max() <= 1.0


def test_similarity_matrix_has_diagonal_values():
    preprocessed = [
        {"tokens": ["kucing", "hitam"]},
        {"tokens": ["kucing", "putih"]},
    ]
    vocab = sorted({token for item in preprocessed for token in item["tokens"]})
    binary_df = create_binary_matrix(preprocessed, vocab)
    matrix = build_similarity_matrix(binary_df)
    assert np.isclose(matrix.iloc[0, 0], 1.0)
    assert np.isclose(matrix.iloc[1, 1], 1.0)
    assert 0 <= matrix.iloc[0, 1] <= 1
