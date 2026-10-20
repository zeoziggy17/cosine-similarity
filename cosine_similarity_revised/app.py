import hashlib
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.file_handler import get_csv_columns, read_text_file, validate_document
from utils.preprocessing import preprocess_texts
from utils.similarity import build_similarity_matrix, build_vocabulary, compute_similarity_pairs, create_binary_matrix


st.set_page_config(page_title="Text Similarity Analyzer", page_icon="📊", layout="wide")

PRIMARY = "#697452"
SECONDARY = "#566044"
BACKGROUND = "#F5F0E6"
PANEL = "#FFFEFA"
SIDEBAR = "#FFFCF7"
BORDER = "#E7D9C4"
TEXT = "#30271F"
SUBTLE = "#8C7B68"
ACCENT = "#B38A58"

SECTIONS = ["input", "preprocessing", "binary", "similarity", "detail"]
SECTION_LABELS = {
    "input": "Input",
    "preprocessing": "Preprocessing",
    "binary": "Binary",
    "similarity": "Similarity",
    "detail": "Detail",
}


def init_state():
    if "texts" not in st.session_state:
        st.session_state.texts = ["", ""]
    if "current_section" not in st.session_state:
        st.session_state.current_section = "input"
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None
    if "selected_csv_column" not in st.session_state:
        st.session_state.selected_csv_column = None
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "manual"


def set_custom_css():
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {BACKGROUND};
            --panel: {PANEL};
            --sidebar: {SIDEBAR};
            --primary: {PRIMARY};
            --primary-dark: {SECONDARY};
            --border: {BORDER};
            --text: {TEXT};
            --muted: {SUBTLE};
            --accent: {ACCENT};
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background: var(--sidebar);
            border-right: 1px solid var(--border);
            box-shadow: 1px 0 0 rgba(48,39,31,0.04);
            width: 290px;
            max-width: 290px;
            min-width: 290px;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding: 1.1rem 1rem 0.8rem 1rem;
        }}
        .app-shell {{
            padding: 0.75rem 0.5rem 2rem 0.5rem;
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }}
        .main-card, .stage-card, .summary-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 6px 18px rgba(48, 39, 31, 0.04);
        }}
        .stage-shell {{
            background: rgba(255,255,255,0.32);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.1rem 1.2rem 1.3rem 1.2rem;
            margin-top: 0.75rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }}
        .header-card {{
            background: rgba(255,255,255,0.35);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 22px rgba(48,39,31,0.03);
        }}
        .badge {{
            display: inline-block;
            background: var(--primary);
            color: white;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.35rem 0.8rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            box-shadow: 0 2px 8px rgba(105,116,82,0.18);
        }}
        .mini-label {{
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.45rem;
        }}
        .app-title {{
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 700;
            line-height: 1.1;
            color: var(--text);
            margin: 0;
        }}
        .section-title {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: var(--text);
        }}
        .section-label {{
            color: var(--primary);
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.68rem;
            font-weight: 800;
            margin-bottom: 0.38rem;
        }}
        .muted {{
            color: var(--muted);
        }}
        .kpi-box {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }}
        .stage-nav {{
            display: flex;
            gap: 0.65rem;
            margin: 0.5rem 0 1.2rem 0;
            flex-wrap: wrap;
        }}
        .stage-nav .stButton > button {{
            background: rgba(255,255,255,0.7);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text);
            font-weight: 600;
            padding: 0.7rem 1.15rem;
            min-height: 46px;
            box-shadow: none;
            transition: all 0.2s ease;
            letter-spacing: 0.02em;
            text-transform: none;
            font-size: 0.88rem;
        }}
        .stage-nav .stButton > button:hover {{
            background: #f7f0e7;
            border-color: var(--primary);
            color: var(--primary-dark);
        }}
        .stage-nav .stButton > button:focus {{
            box-shadow: 0 0 0 0.14rem rgba(105,116,82,0.25);
        }}
        .stage-nav .stButton > button[key="nav-input"],
        .stage-nav .stButton > button[key="nav-preprocessing"],
        .stage-nav .stButton > button[key="nav-binary"],
        .stage-nav .stButton > button[key="nav-similarity"],
        .stage-nav .stButton > button[key="nav-detail"] {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            box-shadow: 0 6px 16px rgba(105,116,82,0.18);
        }}
        .sidebar-box {{
            background: rgba(255,255,255,0.28);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            margin: 0.5rem 0 0.9rem 0;
        }}
        .sidebar-section {{
            padding: 0.15rem 0 0.35rem 0;
            border-bottom: 1px solid rgba(231,217,196,0.9);
            margin-bottom: 0.7rem;
            line-height: 1.55;
            color: var(--text);
            font-size: 0.9rem;
        }}
        .sidebar-section:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}
        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.2rem 0 0.8rem 0;
        }}
        .sidebar-title {{
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            color: var(--text);
        }}
        .sidebar-subtitle {{
            font-size: 0.66rem;
            color: var(--muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .stButton > button {{
            border-radius: 12px;
            background: var(--primary);
            color: white;
            border: 1px solid var(--primary);
            font-weight: 600;
            min-height: 38px;
            padding: 0.55rem 0.9rem;
        }}
        .stButton > button:hover {{
            background: var(--primary-dark);
            border-color: var(--primary-dark);
            color: white;
        }}
        .secondary-button > button {{
            background: white;
            color: var(--primary);
            border: 1px solid var(--primary);
        }}
        .secondary-button > button:hover {{
            background: #f7f1e8;
            color: var(--primary-dark);
        }}
        [data-testid="stFileUploaderDropzone"] {{
            border: 1px dashed var(--border);
            border-radius: 14px;
            background: rgba(255,255,255,0.25);
        }}
        @media (max-width: 768px) {{
            [data-testid="stSidebar"] {{
                width: 100%;
                max-width: none;
                min-width: 0;
            }}
            .app-shell {{
                padding: 0.5rem 0.25rem 2rem 0.25rem;
            }}
            .header-card {{
                padding: 1rem;
            }}
        }}
        .logo-box {{
            width: 36px;
            height: 36px;
            background: var(--accent);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1rem;
        }}
        .sidebar-title {{
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: var(--text);
        }}
        .sidebar-subtitle {{
            font-size: 0.7rem;
            color: var(--muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        [data-testid="stDataFrameResizableWrapper"], [data-testid="stDataFrame"] {{
            overflow-x: auto;
        }}
        .stTextInput input, .stTextArea textarea, .stSelectbox div {{
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .stButton > button {{
            border-radius: 12px;
            background: var(--primary);
            color: white;
            border: 1px solid var(--primary);
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background: var(--primary-dark);
            border-color: var(--primary-dark);
            color: white;
        }}
        .secondary-button > button {{
            background: white;
            color: var(--primary);
            border: 1px solid var(--primary);
        }}
        .secondary-button > button:hover {{
            background: #f7f1e8;
            color: var(--primary-dark);
        }}
        [data-testid="stFileUploaderDropzone"] {{
            border: 1px dashed var(--border);
            border-radius: 14px;
            background: rgba(255,255,255,0.25);
        }}
        @media (max-width: 768px) {{
            .app-shell {{
                padding: 0.5rem 0.25rem 2rem 0.25rem;
            }}
            .header-card {{
                padding: 1rem;
            }}
        }}
        /* Final visual refinements: keep the shell compact and the workspace calm. */
        section[data-testid="stSidebar"] {{
            width: 220px !important;
            min-width: 220px !important;
            max-width: 220px !important;
        }}
        .header-card {{
            background: #F8F4EC;
            padding: 1.25rem 1.55rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 3px 12px rgba(48,39,31,0.025);
        }}
        .stage-shell {{
            background: var(--panel);
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            margin-top: 0.15rem;
            box-shadow: 0 4px 16px rgba(48,39,31,0.035);
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel);
            border-color: var(--border);
            border-radius: 16px;
            padding: 0.5rem;
            box-shadow: 0 4px 16px rgba(48,39,31,0.035);
        }}
        div[role="radiogroup"] {{
            gap: 0.55rem;
            margin: 0.35rem 0 0.85rem;
        }}
        div[role="radiogroup"] label {{
            border: 1px solid var(--border);
            border-radius: 11px;
            background: rgba(255,255,255,0.78);
            padding: 0.55rem 1rem;
            transition: background-color 150ms ease, border-color 150ms ease;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}
        div[role="radiogroup"] label:has(input:checked) p {{
            color: white;
            font-weight: 700;
        }}
        .stButton > button {{
            border-radius: 10px;
            background: #FFFEFA;
            color: var(--text);
            border-color: var(--border);
            font-weight: 600;
            transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
        }}
        .stButton > button:hover {{
            background: #F3EEE4;
            border-color: var(--primary);
            color: var(--primary-dark);
        }}
        .stButton > button[kind="primary"] {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}
        .stButton > button[kind="primary"]:hover {{
            background: var(--primary-dark);
            border-color: var(--primary-dark);
            color: white;
        }}
        .stDownloadButton > button {{
            border-radius: 10px;
            background: #FFFEFA;
            color: var(--primary-dark);
            border: 1px solid var(--border);
        }}
        .stDownloadButton > button:hover {{
            border-color: var(--primary);
            background: #F3EEE4;
            color: var(--primary-dark);
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow-x: auto;
        }}
        .stTextArea textarea, .stTextInput input {{
            background: #FFFEFA;
        }}
        @media (max-width: 768px) {{
            section[data-testid="stSidebar"] {{
                width: min(220px, 86vw) !important;
                min-width: min(220px, 86vw) !important;
                max-width: min(220px, 86vw) !important;
            }}
            .block-container {{
                padding: 1rem 0.75rem 2rem;
            }}
            .app-title {{
                font-size: clamp(1.75rem, 8vw, 2.4rem);
            }}
            div[role="radiogroup"] {{
                overflow-x: auto;
                flex-wrap: nowrap;
                padding-bottom: 0.35rem;
            }}
            div[role="radiogroup"] label {{
                white-space: nowrap;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_all_data():
    keys = [
        "texts",
        "analysis",
        "current_section",
        "last_uploaded_name",
        "last_uploaded_signature",
        "selected_csv_column",
        "input_mode",
        "show_confirmation",
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    for key in list(st.session_state.keys()):
        if key.startswith("sentence_"):
            del st.session_state[key]
    st.session_state.texts = ["", ""]
    st.session_state.current_section = "input"
    st.session_state.analysis = None
    st.session_state.last_uploaded_name = None
    st.session_state.last_uploaded_signature = None
    st.session_state.selected_csv_column = None
    st.session_state.input_mode = "manual"
    st.session_state.show_confirmation = False


def replace_texts(texts: List[str]):
    st.session_state.texts = list(texts)
    for index, text in enumerate(texts):
        st.session_state[f"sentence_{index}"] = text
    for index in range(len(texts), len(texts) + 50):
        key = f"sentence_{index}"
        if key not in st.session_state:
            break
        del st.session_state[key]


def delete_sentence(index: int):
    texts = list(st.session_state.texts)
    del texts[index]
    replace_texts(texts if len(texts) >= 2 else [*texts, ""])
    st.session_state.analysis = None


def build_example_data() -> List[str]:
    return [
        "Sistem informasi manajemen data sangat membantu organisasi dalam memproses data harian.",
        "Organisasi memerlukan sistem informasi manajemen untuk memproses data harian dengan efisien.",
        "Kebijakan publik harus memperhatikan kebutuhan masyarakat dan perbaikan layanan dasar.",
        "Masyarakat membutuhkan layanan dasar yang lebih baik serta kebijakan publik yang responsif.",
        "Teknologi AI membantu perusahaan menganalisis tren pasar dan mengambil keputusan dengan cepat.",
    ]


def validate_texts(texts: List[str]) -> Optional[str]:
    cleaned = [text.strip() for text in texts if str(text).strip()]
    if len(cleaned) < 2:
        return "Minimal dua teks harus diisi untuk melakukan analisis."
    return None


def run_analysis():
    texts = [str(text).strip() for text in st.session_state.texts if str(text).strip()]
    message = validate_texts(texts)
    if message:
        st.warning(message)
        return None

    preprocessed = preprocess_texts(texts)
    vocabulary = build_vocabulary(preprocessed)
    if not vocabulary:
        st.warning(
            "Tidak ada kata yang dapat dianalisis. Masukkan teks yang mengandung huruf atau angka, "
            "bukan hanya spasi atau tanda baca."
        )
        st.session_state.analysis = None
        return None

    binary_df = create_binary_matrix(preprocessed, vocabulary)
    table = compute_similarity_pairs(binary_df)
    matrix = build_similarity_matrix(binary_df)

    analysis = {
        "texts": texts,
        "preprocessed": preprocessed,
        "vocabulary": vocabulary,
        "binary_df": binary_df,
        "pairs_df": table,
        "matrix_df": matrix,
    }
    st.session_state.analysis = analysis
    st.success("Analisis berhasil dijalankan.")
    return analysis


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="logo-box">T</div>
                <div>
                    <div class="sidebar-title">TEXT SIMILARITY ANALYZER</div>
                    <div class="sidebar-subtitle">ANALISIS KEMIRIPAN TEKS</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-box">
                <div style="font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; color:#8C7B68; font-weight:700;">
                    Metode Analisis
                </div>
            </div>
            <div class="sidebar-section">
                <strong>Preprocessing</strong><br>
                <span style="color:#8C7B68;">Case folding<br>Hapus tanda baca<br>Tokenisasi</span>
            </div>
            <div class="sidebar-section">
                <strong>Representasi</strong><br>
                <span style="color:#8C7B68;">Binary Vector (0/1)</span>
            </div>
            <div class="sidebar-section">
                <strong>Similarity</strong><br>
                <span style="color:#8C7B68;">Cosine Similarity</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        if st.button("Reset Data", use_container_width=True, key="reset_button"):
            st.session_state.show_confirmation = True

        if st.session_state.get("show_confirmation"):
            st.warning("Hapus semua input dan hasil analisis?")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Hapus", key="confirm_reset", type="primary", use_container_width=True):
                reset_all_data()
                st.rerun()
            if cancel_col.button("Batal", key="cancel_reset", use_container_width=True):
                st.session_state.show_confirmation = False

        st.markdown(
            "<div style='margin-top:1.2rem; padding-top:0.8rem; border-top:1px solid #E7D9C4; color:#8C7B68; font-size:0.72rem; line-height:1.5;'>Analisis teks berbasis Binary Vector</div>",
            unsafe_allow_html=True,
        )


def render_header():
    st.markdown(
        """
        <div class="header-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
                <div>
                    <div class="mini-label">TEXT ANALYSIS WORKSPACE</div>
                    <h1 class="app-title" style="margin-top: 0.7rem;">Text Similarity Analyzer</h1>
                </div>
                <div class="badge">Text Analysis</div>
            </div>
            <div class="muted" style="margin-top: 0.8rem;">Analisis kemiripan kalimat dengan Binary Representation dan Cosine Similarity.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_nav():
    st.radio(
        "Tahapan analisis",
        options=SECTIONS,
        format_func=lambda section: SECTION_LABELS[section],
        horizontal=True,
        key="current_section",
        label_visibility="collapsed",
    )


def render_stage_panel(title: str, subtitle: str, number: str):
    st.markdown(
        f"""
        <div class='stage-shell'>
            <div class='section-label'>{number}</div>
            <div class='section-title' style='font-size:1.5rem; margin:0;'>{title}</div>
            <div class='muted' style='margin-top:0.35rem;'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_page():
    render_stage_panel("Masukkan kalimat untuk dibandingkan", "Input manual atau unggah file teks untuk memulai analisis.", "SUMBER DATA")

    actions = st.columns([1, 1.15, 1.05, 1.35, 1.05])
    if actions[0].button("Input Manual", key="manual_action", use_container_width=True):
        st.session_state.input_mode = "manual"
        replace_texts(st.session_state.texts)
    if actions[1].button("Upload Dokumen", key="upload_action", use_container_width=True):
        st.session_state.input_mode = "upload"
    if actions[2].button("Tambah Kalimat", key="add_sentence", use_container_width=True):
        replace_texts([*st.session_state.texts, ""])
        st.session_state.input_mode = "manual"
        st.session_state.analysis = None
    if actions[3].button("Gunakan Contoh Data", key="use_example", use_container_width=True):
        replace_texts(build_example_data())
        st.session_state.input_mode = "manual"
        st.session_state.analysis = None
    if actions[4].button("Hapus Input", key="clear_input", use_container_width=True):
        replace_texts(["", ""])
        st.session_state.analysis = None
        st.session_state.input_mode = "manual"

    if st.session_state.input_mode == "upload":
        uploaded_file = st.file_uploader(
            "Unggah file .txt, .csv, .docx, atau .pdf",
            type=["txt", "csv", "docx", "pdf"],
            key="doc_upload",
        )
        if uploaded_file is not None:
            file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            selected_column = None
            if uploaded_file.name.lower().endswith(".csv"):
                try:
                    columns = get_csv_columns(uploaded_file)
                except Exception as exc:
                    columns = []
                    st.error(f"CSV tidak dapat dibaca. Periksa format dan header file. Detail: {exc}")
                if columns:
                    selected_column = st.selectbox(
                        "Pilih kolom teks CSV",
                        options=columns,
                        key=f"csv_column_{file_signature[:12]}",
                    )
            signature = f"{file_signature}:{selected_column or ''}"
            if signature != st.session_state.get("last_uploaded_signature"):
                if not validate_document(uploaded_file.name):
                    st.error("Format file tidak didukung. Gunakan .txt, .csv, .docx, atau .pdf.")
                else:
                    try:
                        loaded = read_text_file(uploaded_file, selected_column=selected_column)

                        if loaded:
                            replace_texts(loaded)
                            st.session_state.last_uploaded_name = uploaded_file.name
                            st.session_state.last_uploaded_signature = signature
                            st.session_state.selected_csv_column = selected_column
                            st.success(f"{len(loaded)} teks berhasil dimuat dari file {uploaded_file.name}.")
                        else:
                            st.warning("File tidak berisi teks yang dapat diekstrak.")
                            st.session_state.last_uploaded_signature = signature
                    except Exception as exc:
                        st.error(f"Gagal membaca file: {exc}")
            elif st.session_state.texts:
                st.caption(
                    f"{len(st.session_state.texts)} teks termuat. Beralih ke Input Manual untuk meninjau atau mengedit teks."
                )

    if len(st.session_state.texts) < 2:
        replace_texts(["", ""])

    if st.session_state.input_mode == "manual":
        for idx, text in enumerate(list(st.session_state.texts)):
            with st.container(border=True):
                text_col, delete_col = st.columns([12, 1])
                delete_col.button(
                    "×",
                    key=f"delete_sentence_{idx}",
                    help=f"Hapus teks S{idx + 1}",
                    on_click=delete_sentence,
                    args=(idx,),
                )
                with text_col:
                    widget_key = f"sentence_{idx}"
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = text
                    st.text_area(
                        f"S{idx + 1}",
                        key=widget_key,
                        height=90,
                        placeholder=f"Masukkan teks kalimat S{idx + 1}...",
                        label_visibility="visible",
                    )

        st.session_state.texts = [
            st.session_state.get(f"sentence_{idx}", "")
            for idx in range(len(st.session_state.texts))
        ]

    if st.session_state.analysis is not None:
        current_texts = [
            str(text).strip()
            for text in st.session_state.texts
            if str(text).strip()
        ]
        if current_texts != st.session_state.analysis["texts"]:
            st.session_state.analysis = None

    if st.button("Analisis Kemiripan", key="run_similarity", type="primary"):
        run_analysis()


def render_preprocessing_page():
    render_stage_panel("Preprocessing & tokenisasi", "Case folding, penghapusan tanda baca, dan pemecahan kata untuk analisis selanjutnya.", "TAHAP 01")
    if st.session_state.analysis is None:
        st.info("Silakan jalankan analisis terlebih dahulu pada halaman Input.")
        return

    texts = st.session_state.analysis["texts"]
    preprocessed = st.session_state.analysis["preprocessed"]

    df = pd.DataFrame(
        [
            {
                "ID Teks": f"S{index + 1}",
                "Teks Asli": item["original"],
                "Case Folding": item["case_folded"],
                "Penghapusan Tanda Baca": item["without_punctuation"],
                "Tokenisasi": ", ".join(item["tokens"]) if item["tokens"] else "(kosong)",
            }
            for index, item in enumerate(preprocessed)
        ]
    )
    st.dataframe(df, use_container_width=True)


def render_binary_page():
    render_stage_panel(
        "Binary representation",
        "Nilai 1 menandakan kata muncul pada kalimat; nilai 0 menandakan tidak muncul.",
        "TAHAP 02",
    )
    if st.session_state.analysis is None:
        st.info("Jalankan analisis dengan mengisi minimal dua kalimat pada tab Input.")
        return

    binary_df = st.session_state.analysis["binary_df"]
    vocab = st.session_state.analysis["vocabulary"]

    st.caption(f"Vocabulary berisi {len(vocab)} kata unik.")
    if binary_df.empty or binary_df.shape[1] == 0:
        st.warning("Tidak ada matriks binary karena tidak ada token yang valid.")
        return

    # Tampilkan kata sebagai baris dan teks sebagai kolom, seperti referensi.
    display_binary = binary_df.T.copy()
    display_binary.index.name = "Vocabulary"
    display_binary.columns = [f"S{i + 1}" for i in range(len(display_binary.columns))]
    styled = display_binary.style.map(
        lambda value: "background-color: #DDE7CF; color: #30271F;" if value == 1
        else "background-color: #F5F0E6; color: #8C7B68;"
    )
    st.dataframe(styled, use_container_width=True, height=min(620, 44 + 30 * len(display_binary)))
    st.caption("Nilai 1 = kata muncul, nilai 0 = kata tidak muncul.")
    st.download_button(
        "Unduh Binary Vector (CSV)",
        data=display_binary.to_csv(index=True).encode("utf-8-sig"),
        file_name="binary_vector.csv",
        mime="text/csv",
    )

def render_similarity_page():
    render_stage_panel(
        "Hasil cosine similarity",
        "Semakin mendekati 1, semakin tinggi kemiripan kata pada pasangan kalimat.",
        "TAHAP 03",
    )
    if st.session_state.analysis is None:
        st.info("Jalankan analisis dengan mengisi minimal dua kalimat pada tab Input.")
        return

    pairs_df = st.session_state.analysis["pairs_df"].copy()
    if pairs_df.empty:
        st.warning("Tidak ada pasangan teks yang dapat dibandingkan.")
        return

    highest_value = float(pairs_df["Persentase"].max())
    top_pairs = pairs_df[pairs_df["Persentase"] == highest_value]
    top_labels = [
        f'{row["Teks Pertama"]} - {row["Teks Kedua"]}'
        for _, row in top_pairs.iterrows()
    ]

    top_card = st.container(border=True)
    with top_card:
        st.markdown("<div class='mini-label'>PASANGAN PALING MIRIP</div>", unsafe_allow_html=True)
        st.markdown(
            f"**{', '.join(top_labels)}** · Cosine Similarity **{highest_value / 100:.4f}**"
        )

    display_df = pd.DataFrame({
        "Ranking": [f"#{i}" for i in range(1, len(pairs_df) + 1)],
        "Pasangan": [
            f'{row["Teks Pertama"]} - {row["Teks Kedua"]}'
            for _, row in pairs_df.iterrows()
        ],
        "Cosine Similarity": pairs_df["Cosine Similarity"].astype(float),
    })
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ranking": st.column_config.TextColumn("Ranking", width="small"),
            "Pasangan": st.column_config.TextColumn("Pasangan", width="medium"),
            "Cosine Similarity": st.column_config.ProgressColumn(
                "Cosine Similarity",
                help="Nilai cosine similarity dari 0 sampai 1.",
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
                width="large",
            ),
        },
    )

    with st.expander("Lihat tabel lengkap hasil kemiripan"):
        full_df = pairs_df.copy()
        full_df["Cosine Similarity"] = full_df["Cosine Similarity"].map(lambda v: f"{v:.4f}")
        full_df["Persentase"] = full_df["Persentase"].map(lambda v: f"{v:.2f}%")
        st.dataframe(full_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Unduh CSV Hasil Cosine Similarity",
        data=pairs_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_cosine_similarity.csv",
        mime="text/csv",
    )

def render_detail_page():
    render_stage_panel(
        "Detail perhitungan",
        "Matriks cosine similarity dan contoh uraian perhitungan pasangan teratas.",
        "TAHAP 04",
    )
    if st.session_state.analysis is None:
        st.info("Jalankan analisis dengan mengisi minimal dua kalimat pada tab Input.")
        return

    analysis = st.session_state.analysis
    binary_df = analysis["binary_df"]
    matrix_df = analysis["matrix_df"]
    pairs_df = analysis["pairs_df"]

    st.markdown(
        "<div class='formula-card'><strong>cos(A, B) = (A · B) / (||A|| × ||B||)</strong>"
        "<div class='muted' style='margin-top:0.4rem;'>Vektor dibangun dari vocabulary bersama. "
        "Setiap kata bernilai 1 jika ada pada kalimat, selain itu 0.</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Similarity matrix")
    matrix_display = (matrix_df * 1.0).copy()
    matrix_display.index.name = "Teks"
    st.dataframe(
        matrix_display.style.format("{:.4f}").background_gradient(
            cmap="YlGn", vmin=0, vmax=1
        ),
        use_container_width=True,
    )

    if not pairs_df.empty:
        best_pair = pairs_df.iloc[0]
        first_label = str(best_pair["Teks Pertama"])
        second_label = str(best_pair["Teks Kedua"])
        first_index = int(first_label[1:]) - 1
        second_index = int(second_label[1:]) - 1
        vector_a = binary_df.iloc[first_index].to_numpy(dtype=float)
        vector_b = binary_df.iloc[second_index].to_numpy(dtype=float)
        common_words = sorted(set(binary_df.columns[vector_a.astype(bool) & vector_b.astype(bool)]))
        dot_product = float(np.dot(vector_a, vector_b))
        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        similarity = float(best_pair["Cosine Similarity"])

        with st.container(border=True):
            st.markdown(
                f"**Contoh perhitungan untuk {first_label} – {second_label}**"
            )
            st.caption(f"Dot product = {dot_product:g} · ||A|| = {norm_a:.4f} · ||B|| = {norm_b:.4f}")
            st.caption(
                "Kata unik yang sama: " + (", ".join(common_words) if common_words else "(tidak ada)")
            )
            st.latex(
                rf"\cos(A,B)=\frac{{{dot_product:g}}}{{{norm_a:.4f}\times {norm_b:.4f}}}"
                rf"={similarity:.4f}"
            )
            st.markdown(f"**Cosine Similarity:** {similarity:.4f} · **Kemiripan:** {similarity * 100:.2f}%")

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Unduh Binary Vector",
        data=binary_df.T.to_csv(index=True).encode("utf-8-sig"),
        file_name="binary_vector.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Unduh Matriks Similarity",
        data=matrix_df.to_csv(index=True).encode("utf-8-sig"),
        file_name="matriks_similarity.csv",
        mime="text/csv",
        use_container_width=True,
    )

def main():
    init_state()
    set_custom_css()
    render_sidebar()
    render_header()
    render_stage_nav()
    with st.container(border=True):
        section = st.session_state.current_section
        if section == "input":
            render_input_page()
        elif section == "preprocessing":
            render_preprocessing_page()
        elif section == "binary":
            render_binary_page()
        elif section == "similarity":
            render_similarity_page()
        elif section == "detail":
            render_detail_page()


if __name__ == "__main__":
    main()
