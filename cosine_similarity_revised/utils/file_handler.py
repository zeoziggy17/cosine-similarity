from io import BytesIO
from typing import List, Optional

import pandas as pd


SUPPORTED_EXTENSIONS = {"txt", "csv", "docx", "pdf"}


def get_file_extension(file_name: str) -> str:
    return file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""


def validate_document(file_name: str) -> bool:
    return get_file_extension(file_name) in SUPPORTED_EXTENSIONS


def read_text_file(file, selected_column: Optional[str] = None) -> List[str]:
    if file is None:
        return []
    extension = get_file_extension(file.name)
    file_value = file.getvalue()

    if extension == "txt":
        text = file_value.decode("utf-8-sig", errors="replace")
        return [line.strip() for line in text.splitlines() if line.strip()]

    if extension == "csv":
        dataframe = pd.read_csv(BytesIO(file_value), encoding="utf-8-sig")
        if dataframe.empty:
            return []
        if selected_column is None:
            text_columns = [column for column in dataframe.columns if dataframe[column].map(lambda value: isinstance(value, str)).any()]
            selected_column = text_columns[0] if text_columns else dataframe.columns[0]
        values = dataframe[selected_column].dropna().astype(str)
        return [str(value).strip() for value in values if str(value).strip()]

    if extension == "docx":
        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Paket python-docx tidak tersedia. Instal dependency yang diperlukan.") from exc
        document = Document(BytesIO(file_value))
        texts = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                texts.append(text)
        return texts

    if extension == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Paket pypdf tidak tersedia. Instal dependency yang diperlukan.") from exc
        reader = PdfReader(BytesIO(file_value))
        extracted_texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_texts.append(page_text.strip())
        return extracted_texts

    raise ValueError(f"Format file '{extension}' tidak didukung.")


def get_csv_columns(file) -> List[str]:
    if file is None:
        return []
    extension = get_file_extension(file.name)
    if extension != "csv":
        return []
    dataframe = pd.read_csv(BytesIO(file.getvalue()), encoding="utf-8-sig")
    return list(dataframe.columns)
