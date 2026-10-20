from typing import List, Dict

import numpy as np
import pandas as pd


def build_vocabulary(preprocessed_texts: List[Dict[str, object]]) -> List[str]:
    vocab = sorted({token for item in preprocessed_texts for token in item.get("tokens", [])})
    return vocab


def create_binary_matrix(preprocessed_texts: List[Dict[str, object]], vocabulary: List[str]) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(preprocessed_texts):
        tokens = item.get("tokens", [])
        binary_row = [1 if token in tokens else 0 for token in vocabulary]
        rows.append(binary_row)
    labels = [f"S{index + 1}" for index in range(len(preprocessed_texts))]
    return pd.DataFrame(rows, index=labels, columns=vocabulary, dtype=int)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    dot_product = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0.0
    return dot_product / denominator


def compute_similarity_pairs(binary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = binary_df.index.tolist()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            left = binary_df.iloc[i].to_numpy(dtype=float)
            right = binary_df.iloc[j].to_numpy(dtype=float)
            common_words = int(np.dot(left, right))
            cosine_value = cosine_similarity(left, right)
            rows.append(
                {
                    "No": len(rows) + 1,
                    "Teks Pertama": labels[i],
                    "Teks Kedua": labels[j],
                    "Jumlah Kata Sama": common_words,
                    "Cosine Similarity": cosine_value,
                    "Persentase": cosine_value * 100,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("Persentase", ascending=False, kind="mergesort").reset_index(drop=True)
    table["No"] = range(1, len(table) + 1)
    table["Kategori"] = table["Persentase"].apply(classify_similarity)
    return table


def classify_similarity(percent: float) -> str:
    if percent == 0:
        return "0%: Tidak ada kemiripan berdasarkan fitur kata."
    if 0 < percent < 30:
        return "Rendah"
    if 30 <= percent < 60:
        return "Sedang"
    if 60 <= percent < 80:
        return "Tinggi"
    return "Sangat tinggi"


def build_similarity_matrix(binary_df: pd.DataFrame) -> pd.DataFrame:
    labels = binary_df.index.tolist()
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for i in range(len(labels)):
        for j in range(len(labels)):
            vec_i = binary_df.iloc[i].to_numpy(dtype=float)
            vec_j = binary_df.iloc[j].to_numpy(dtype=float)
            if i == j:
                matrix[i, j] = 1.0 if np.any(vec_i) else 0.0
            else:
                value = cosine_similarity(vec_i, vec_j)
                matrix[i, j] = value
                matrix[j, i] = value
    return pd.DataFrame(matrix, index=labels, columns=labels)
