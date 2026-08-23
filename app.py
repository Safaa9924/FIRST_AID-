"""
app.py — Nabda | نَبْضَة (Streamlit)

Before running this app you MUST run build_index.py once locally to
generate the data/ folder (and pdf_images/ if you want page images).
This app only loads the already-built index — it does not process
the PDF itself.

LLM key: read ONLY from st.secrets / environment. There is no input
field for it anywhere in the UI. If it's missing, the site shows a
generic "service unavailable" message with no technical detail.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from rag_core import (
    CONFIG, MiniBM25,  # noqa: F401  (MiniBM25 must be imported so pickle can find the class)
    retrieve_top_k_hybrid, rerank_candidates,
    build_context_package, build_chat_prompt, generate_answer,
    correct_user_query, detect_language, translate_to_english, translate_to_arabic,
    expand_query, confidence_label,
)

SITE_NAME = "نَبْضَة"
SITE_TAGLINE = "Every heartbeat of hesitation costs a life"

DATA_DIR = CONFIG["DATA_DIR"]
TOP_K = CONFIG["TOP_K"]
TOP_N_RERANK = CONFIG["TOP_N_RERANK"]

st.set_page_config(page_title=f"{SITE_NAME} | Nabda", page_icon="🩺", layout="centered")


# ======================================================================
# Backend: load the pre-built index + models (cached so it loads once)
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_index():
    missing = [
        f for f in ["chunks_df.pkl", "tfidf_index.pkl", "bm25_index.pkl",
                     "embedding_matrix.npy", "page_to_images.pkl"]
        if not os.path.exists(os.path.join(DATA_DIR, f))
    ]
    if missing:
        return None

    chunks_df = pd.read_pickle(os.path.join(DATA_DIR, "chunks_df.pkl"))

    with open(os.path.join(DATA_DIR, "tfidf_index.pkl"), "rb") as f:
        tfidf_data = pickle.load(f)

    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)

    embedding_matrix = np.load(os.path.join(DATA_DIR, "embedding_matrix.npy"))

    with open(os.path.join(DATA_DIR, "page_to_images.pkl"), "rb") as f:
        page_to_images = pickle.load(f)

    return {
        "chunks_df": chunks_df,
        "tfidf_vectorizer": tfidf_data["vectorizer"],
        "tfidf_matrix": tfidf_data["matrix"],
        "bm25": bm25,
        "embedding_matrix": embedding_matrix,
        "page_to_images": page_to_images,
    }


@st.cache_resource(show_spinner=False)
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedding_model = SentenceTransformer(CONFIG["EMBEDDING_MODEL_NAME"])
    reranker = CrossEncoder(CONFIG["RERANKER_MODEL_NAME"])
    return embedding_model, reranker


# ---------------------------------------------------------------
# API Key: بتتقرا من secrets فقط. مفيش أي input أو ذكر ليها
# في الواجهة نهائيًا. لو مش موجودة، الموقع بيوريك رسالة صيانة
# عادية من غير أي تفاصيل تقنية.
# ---------------------------------------------------------------
def _get_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


# ======================================================================
# UI: تنسيق (CSS) عشان الموقع يبقى شكله منصة إسعافات أولية حقيقية
# ======================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@700;800;900&family=Tajawal:wght@400;500;700&display=swap');

    :root {
        --ink: #0E2A24;
        --bg: #F5FAF8;
        --surface: #FFFFFF;
        --teal: #0B7A66;
        --teal-deep: #06463C;
        --coral: #FF5A4E;
        --amber: #F2A93B;
        --line: #DCEEE8;
    }

    html, body, [class^="css"], [class*=" css"] {
        font-family: 'Tajawal', sans-serif;
        color: var(--ink);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp { background: var(--bg); }

    .block-container { padding-top: 1.4rem; max-width: 760px; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        background: var(--teal-deep);
        background-image:
            radial-gradient(circle at 12% 15%, rgba(11,122,102,0.65), transparent 42%),
            radial-gradient(circle at 88% 85%, rgba(255,90,78,0.22), transparent 45%);
        border-radius: 24px;
        padding: 2.6rem 1.6rem 2.1rem;
        text-align: center;
        margin-bottom: 1.8rem;
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(6,70,60,0.28);
    }
    .hero-eyebrow {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        color: var(--amber);
        font-weight: 700;
        text-transform: uppercase;
    }
    .hero h1 {
        font-family: 'Cairo', sans-serif;
        font-weight: 900;
        font-size: 2.8rem;
        color: #F5FAF8;
        margin: 0.25rem 0 0.15rem;
        letter-spacing: -0.01em;
    }
    .hero .tagline {
        font-family: 'Tajawal', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #CDEDE4;
        margin-bottom: 0.9rem;
    }
    .pulse-wrap { width: 100%; max-width: 380px; margin: 0 auto 0.9rem; }
    .pulse-line {
        stroke: var(--coral);
        stroke-width: 2.5;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 900;
        stroke-dashoffset: 900;
        animation: draw 2.8s ease-in-out infinite;
    }
    @keyframes draw {
        0%   { stroke-dashoffset: 900; opacity: 0.35; }
        55%  { stroke-dashoffset: 0;   opacity: 1; }
        100% { stroke-dashoffset: -900; opacity: 0.35; }
    }
    .hero p.sub {
        font-size: 0.92rem;
        color: #A9D6CB;
        max-width: 460px;
        margin: 0 auto 1.1rem;
        line-height: 1.7;
    }
    .chip-row { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
    .info-chip {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        color: #EAFBF6;
        border-radius: 999px;
        padding: 0.35rem 0.95rem;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 1.15rem;
        color: var(--teal-deep);
        margin: 0.2rem 0 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---------- Topic cards ---------- */
    .category-card {
        background: var(--surface);
        border-radius: 16px;
        padding: 1.1rem 0.5rem 0.9rem;
        text-align: center;
        border: 1px solid var(--line);
        box-shadow: 0 2px 10px rgba(6,70,60,0.05);
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .category-card:hover {
        transform: translateY(-3px);
        border-color: var(--teal);
        box-shadow: 0 10px 22px rgba(6,70,60,0.14);
    }
    .category-card .emoji {
        font-size: 1.5rem;
        width: 44px; height: 44px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.5rem;
        background: var(--bg);
        border: 1px solid var(--line);
        border-radius: 50%;
    }
    .category-card .label {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink);
    }

    /* ---------- Disclaimer ---------- */
    .disclaimer {
        background: #FFF6E8;
        border: 1px solid #F3D998;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.83rem;
        color: #7A5200;
        margin-top: 0.9rem;
        line-height: 1.7;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        border-radius: 10px !important;
        font-family: 'Tajawal', sans-serif !important;
        font-weight: 700 !important;
        border: 1px solid var(--line) !important;
        color: var(--teal-deep) !important;
    }
    .stButton>button:hover {
        border-color: var(--teal) !important;
        color: var(--teal) !important;
        background: #F0FAF7 !important;
    }
    .stButton>button[kind="primary"] {
        background: var(--coral) !important;
        border-color: var(--coral) !important;
        color: white !important;
    }

    /* ---------- Chat ---------- */
    section[data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid var(--line);
        background: var(--surface);
        box-shadow: 0 1px 6px rgba(6,70,60,0.04);
    }
    div[data-testid="stChatInput"] textarea {
        font-family: 'Tajawal', sans-serif !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--teal-deep);
        background-image: radial-gradient(circle at 20% 0%, rgba(11,122,102,0.55), transparent 55%);
    }
    section[data-testid="stSidebar"] * { color: #EAFBF6; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

    .side-brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 1.4rem;
    }
    .side-brand .dot {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: var(--coral);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .side-brand .name {
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        color: #F5FAF8;
    }
    .side-brand .role {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.72rem;
        color: #9FD9CB;
        font-weight: 500;
    }

    .side-heading {
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 0.95rem;
        color: #F5FAF8 !important;
        margin: 1.1rem 0 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .side-text {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.85rem;
        color: #BFE7DC !important;
        line-height: 1.7;
    }
    .side-divider {
        height: 1px;
        background: rgba(255,255,255,0.14);
        border: none;
        margin: 1.1rem 0;
    }

    section[data-testid="stSidebar"] .disclaimer {
        background: rgba(242,169,59,0.14);
        border: 1px solid rgba(242,169,59,0.4);
    }
    section[data-testid="stSidebar"] .disclaimer * { color: #FBE3AE !important; }

    section[data-testid="stSidebar"] .stButton>button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        color: #EAFBF6 !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: var(--coral) !important;
        border-color: var(--coral) !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] label { color: #EAFBF6 !important; }
</style>
"""

PULSE_SVG = (
    '<div class="pulse-wrap"><svg viewBox="0 0 380 60" xmlns="http://www.w3.org/2000/svg">'
    '<path class="pulse-line" d="M0,32 L70,32 L88,10 L104,52 L120,20 L138,32 L190,32 '
    'L208,10 L224,52 L240,20 L258,32 L380,32" /></svg></div>'
)

FIRST_AID_TOPICS = [
    ("🩸", "نزيف", "ما هي الخطوات الصحيحة للسيطرة على النزيف وإسعافه؟"),
    ("🔥", "حروق", "كيف يتم التعامل الصحيح مع إصابات الحروق؟"),
    ("🫁", "اختناق", "الاختناق"),
    ("❤️", "إنعاش قلبي (CPR)", "ما هي خطوات الإنعاش القلبي الرئوي (CPR) الصحيحة؟"),
    ("🦴", "كسور", "كيف أتعامل مع حالة اشتباه في وجود كسر؟"),
    ("🐝", "لسعات وحساسية", "ما هو الإسعاف الأولي للسعات الحشرات وردود الفعل التحسسية؟"),
]


def render_sources(sources_payload):
    """جدول بالمصادر: الترتيب، القسم، الصفحة، الدرجة، الثقة — زي النسخة الأولى.
    وبيعرض صورة الصفحة كمان لو كانت متاحة."""
    if not sources_payload:
        return
    table_rows = []
    for i, s in enumerate(sources_payload, start=1):
        table_rows.append({
            "الترتيب": i,
            "القسم": s["section"],
            "الصفحة": s["page_number"] if s["page_number"] is not None else "—",
            "الدرجة": round(s["score"], 2) if s["score"] is not None else "—",
            "الثقة": s["confidence"],
        })
    sources_df = pd.DataFrame(table_rows)
    st.dataframe(sources_df, hide_index=True, use_container_width=True)

    with st.expander("📄 نص المقاطع كاملًا"):
        for i, s in enumerate(sources_payload, start=1):
            st.markdown(f"**{i}. {s['section']}** — صفحة {s['page_number'] or '—'}")
            st.write(s["text"])

    # صور الصفحات المصدرية (لو موجودة فعليًا وأتاحها build_index.py)
    shown_pages = set()
    image_cols_data = []
    for s in sources_payload:
        page = s["page_number"]
        images = s.get("images") or []
        if page is None or page in shown_pages or not images:
            continue
        shown_pages.add(page)
        img_path = images[0]
        if os.path.exists(img_path):
            image_cols_data.append((img_path, page))

    if image_cols_data:
        with st.expander(f"🖼️ صور الصفحات ({len(image_cols_data)})"):
            cols = st.columns(min(3, len(image_cols_data)))
            for i, (img_path, page) in enumerate(image_cols_data):
                with cols[i % len(cols)]:
                    st.image(img_path, caption=f"صفحة {page}", use_container_width=True)


def render_hero():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    hero_html = (
        '<div class="hero">'
        '<div class="hero-eyebrow">مساعد الطوارئ الرقمي</div>'
        f'<h1>{SITE_NAME}</h1>'
        f'<div class="tagline">{SITE_TAGLINE}</div>'
        f'{PULSE_SVG}'
        '<p class="sub">إجابات فورية وموثوقة وقت الطوارئ، مبنية على مرجع معتمد '
        'من أساسيات الإسعافات الأولية العالمية.</p>'
        '<div class="chip-row">'
        '<span class="info-chip">🌐 واجهة عربية بالكامل</span>'
        '<span class="info-chip">⚡ إجابة فورية</span>'
        '<span class="info-chip">📚 مصادر موثقة</span>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_topics():
    st.markdown('<div class="section-label">🗂️ استكشف الحالات الشائعة</div>', unsafe_allow_html=True)
    cols = st.columns(len(FIRST_AID_TOPICS))
    clicked_question = None
    for col, (emoji, label, topic_question) in zip(cols, FIRST_AID_TOPICS):
        with col:
            card_html = (
                '<div class="category-card">'
                f'<div class="emoji">{emoji}</div>'
                f'<div class="label">{label}</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button("اسأل", key=f"topic_{label}", use_container_width=True):
                clicked_question = topic_question
    return clicked_question


def render_sidebar(chunk_count):
    with st.sidebar:
        brand_html = (
            '<div class="side-brand">'
            '<div class="dot">💓</div>'
            '<div>'
            f'<div class="name">{SITE_NAME}</div>'
            '<div class="role">مساعد الإسعافات الأولية</div>'
            '</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)
        st.markdown(f'<div class="side-heading">💓 عن {SITE_NAME}</div>', unsafe_allow_html=True)
        about_html = (
            f'<p class="side-text">«{SITE_NAME}» منصة إرشادية تقدّم معلومات إسعافات أولية '
            'سريعة وموثوقة، مبنية على مصادر طبية معتمدة '
            f'({chunk_count} مقطع مفهرس من المرجع الرسمي).</p>'
        )
        st.markdown(about_html, unsafe_allow_html=True)
        st.markdown('<hr class="side-divider">', unsafe_allow_html=True)
        st.markdown('<div class="side-heading">🚑 تذكير مهم</div>', unsafe_allow_html=True)
        disclaimer_html = (
            "<div class='disclaimer'>في حالة الطوارئ الحقيقية، اتصل فورًا "
            "بخدمات الإسعاف المحلية. المعلومات هنا للإرشاد الأولي فقط "
            "ولا تغني عن الرعاية الطبية المتخصصة.</div>"
        )
        st.markdown(disclaimer_html, unsafe_allow_html=True)
        st.markdown('<hr class="side-divider">', unsafe_allow_html=True)
        show_sources = st.toggle("📚 عرض المصادر مع كل إجابة", value=False)
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        return show_sources


# ======================================================================
# RAG pipeline (uses the pre-built index from data/ via rag_core)
# ======================================================================
# ======================================================================
# Answer Question
# ======================================================================
def answer_question(question, index, embedding_model, reranker):

    original_question = question

    correction_info = {
        "was_corrected": False,
        "original": original_question,
        "corrected": original_question
    }

    try:

        # ==============================================================
        # 1. Query Correction
        # ==============================================================
        corrected_question, was_corrected = correct_user_query(
            question,
            use_llm=True
        )

        question = corrected_question or original_question

        correction_info = {
            "was_corrected": was_corrected,
            "original": original_question,
            "corrected": question
        }

        # ==============================================================
        # 2. Language Detection
        # ==============================================================
        lang = detect_language(question)

        # ==============================================================
        # 3. Translate Query to English
        # ==============================================================
        if lang != "en":
            retrieval_query = translate_to_english(question)
        else:
            retrieval_query = question

        if not retrieval_query:
            retrieval_query = question

        # ==============================================================
        # 4. Query Expansion
        # ==============================================================
        expanded_query = expand_query(retrieval_query)

        if not expanded_query:
            expanded_query = retrieval_query

        # ==============================================================
        # 5. Hybrid Retrieval
        # ==============================================================
        candidates = retrieve_top_k_hybrid(
            expanded_query,
            index["bm25"],
            embedding_model,
            index["embedding_matrix"],
            index["chunks_df"],
            k=TOP_K,
        )

        if candidates is None or len(candidates) == 0:

            return (
                None,
                None,
                [],
                "no_sources",
                correction_info
            )

        # ==============================================================
        # 6. Reranking
        # ==============================================================
        reranked = rerank_candidates(
            expanded_query,
            candidates,
            reranker,
            top_n=TOP_N_RERANK
        )

        if reranked is None or len(reranked) == 0:

            return (
                None,
                None,
                [],
                "no_sources",
                correction_info
            )

        # ==============================================================
        # 7. Build Context
        # ==============================================================
        context = build_context_package(
            query=expanded_query,
            reranked_df=reranked
        )

        if not context:
            return (
                None,
                None,
                [],
                "no_sources",
                correction_info
            )

        if context.get("num_sources", 0) == 0:

            return (
                None,
                None,
                [],
                "no_sources",
                correction_info
            )

        # ==============================================================
        # 8. Build LLM Prompt
        # ==============================================================
        prompt = build_chat_prompt(
            condition=question,
            context_text=context.get("context_text", "")
        )

        if not prompt:
            return (
                None,
                None,
                [],
                "llm_error",
                correction_info
            )

        # ==============================================================
        # 9. Generate English Answer
        # ==============================================================
        raw = generate_answer(prompt)

        if not raw:

            return (
                "LLM returned an empty response.",
                "LLM returned an empty response.",
                [],
                "llm_error",
                correction_info
            )

        # ==============================================================
        # 10. Handle LLM Errors
        # ==============================================================
        if isinstance(raw, str) and raw.startswith("__LLM_ERROR__"):

            err_details = raw.replace(
                "__LLM_ERROR__:",
                ""
            ).strip()

            return (
                err_details,
                err_details,
                [],
                "llm_error",
                correction_info
            )

        answer_en = raw

        # ==============================================================
        # 11. Translate Answer to Arabic
        # ==============================================================
        try:

            answer_ar = translate_to_arabic(answer_en)

        except Exception as translation_error:

            answer_ar = (
                "تعذر ترجمة الإجابة للعربية.\n\n"
                f"Translation error: {type(translation_error).__name__}: "
                f"{translation_error}"
            )

        # ==============================================================
        # 12. Prepare Sources
        # ==============================================================
        sources_payload = []

        selected_df = context.get("selected_df")

        if selected_df is not None and len(selected_df) > 0:

            for _, row in selected_df.iterrows():

                page = row.get("page_number")
                score = row.get("rerank_score")

                # ------------------------------------------------------
                # Page Number
                # ------------------------------------------------------
                try:

                    page_int = (
                        int(page)
                        if pd.notna(page) and page != -1
                        else None
                    )

                except Exception:

                    page_int = None

                # ------------------------------------------------------
                # Images
                # ------------------------------------------------------
                if page_int is not None:

                    images = index.get(
                        "page_to_images",
                        {}
                    ).get(
                        page_int,
                        []
                    )

                else:

                    images = []

                # ------------------------------------------------------
                # Score
                # ------------------------------------------------------
                try:

                    score_float = (
                        float(score)
                        if pd.notna(score)
                        else None
                    )

                except Exception:

                    score_float = None

                # ------------------------------------------------------
                # Confidence
                # ------------------------------------------------------
                try:

                    confidence = (
                        confidence_label(score)
                        if pd.notna(score)
                        else "—"
                    )

                except Exception:

                    confidence = "—"

                # ------------------------------------------------------
                # Source Payload
                # ------------------------------------------------------
                sources_payload.append({

                    "chunk_id": row.get(
                        "chunk_id",
                        ""
                    ),

                    "section": row.get(
                        "section",
                        "N/A"
                    ),

                    "page_number": page_int,

                    "score": score_float,

                    "confidence": confidence,

                    "text": row.get(
                        "chunk_text",
                        ""
                    ),

                    "images": images,
                })

        # ==============================================================
        # 13. SUCCESS
        # ==============================================================
        return (
            answer_en,
            answer_ar,
            sources_payload,
            None,
            correction_info
        )

    # ==================================================================
    # ERROR HANDLING
    # ==================================================================
    except Exception as e:

        import traceback

        print("=" * 80)
        print("ERROR INSIDE answer_question")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))
        print("TRACEBACK:")
        print(traceback.format_exc())
        print("=" * 80)

        err_msg = (
            f"{type(e).__name__}: {str(e)}"
        )

        return (
            err_msg,
            err_msg,
            [],
            "llm_error",
            correction_info
        )


# ======================================================================
# Main
# ======================================================================
def main():

    render_hero()

    # ==================================================================
    # Load Index
    # ==================================================================
    index = load_index()

    if index is None:

        with st.sidebar:
            pass

        st.error(
            "الخدمة غير متاحة حاليًا، برجاء المحاولة لاحقًا 🙏\n\n"
            "(لمطوّر الموقع: لم يتم العثور على مجلد `data/` — "
            "شغّل `build_index.py` أولًا وارفع نتائجه. "
            "راجع SETUP.md)."
        )

        st.stop()

    # ==================================================================
    # Sidebar
    # ==================================================================
    show_sources = render_sidebar(
        len(index["chunks_df"])
    )

    # ==================================================================
    # API Key
    # ==================================================================
    api_key = _get_api_key()

    if api_key:

        CONFIG["GROQ_API_KEY"] = api_key

    # ==================================================================
    # Chat History
    # ==================================================================
    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    # ==================================================================
    # Load Models
    # ==================================================================
    with st.spinner(
        "جارِ تجهيز قاعدة المعرفة..."
    ):

        embedding_model, reranker = load_models()

    # ==================================================================
    # Topics
    # ==================================================================
    clicked_topic = render_topics()

    st.markdown("---")

    st.markdown(
        '<div class="section-label">💬 اسأل المساعد</div>',
        unsafe_allow_html=True
    )

    # ==================================================================
    # Display Previous Chat
    # ==================================================================
    for msg in st.session_state.chat_history:

        with st.chat_message(
            msg["role"],
            avatar=(
                "🚑"
                if msg["role"] == "assistant"
                else "🧑"
            )
        ):

            if msg["role"] == "assistant":

                if (
                    "content_en" in msg
                    and msg["content_en"]
                ):

                    st.markdown(
                        "**🇬🇧 English**"
                    )

                    st.markdown(
                        msg["content_en"]
                    )

                    st.markdown("---")

                    st.markdown(
                        "**🇪🇬 بالعربي**"
                    )

                    st.markdown(
                        msg.get(
                            "content_ar",
                            ""
                        )
                    )

                else:

                    st.markdown(
                        msg.get(
                            "content",
                            ""
                        )
                    )

            else:

                st.markdown(
                    msg["content"]
                )

            # ----------------------------------------------------------
            # Sources
            # ----------------------------------------------------------
            if (
                msg.get("sources")
                and show_sources
            ):

                st.markdown(
                    "**📚 المصادر**"
                )

                render_sources(
                    msg["sources"]
                )

    # ==================================================================
    # User Input
    # ==================================================================
    question = st.chat_input(
        "اكتب سؤالك عن الإسعافات الأولية هنا..."
    )

    if clicked_topic and not question:

        question = clicked_topic

    # ==================================================================
    # Process Question
    # ==================================================================
    if question:

        # --------------------------------------------------------------
        # Add User Message
        # --------------------------------------------------------------
        st.session_state.chat_history.append({

            "role": "user",

            "content": question

        })

        with st.chat_message(
            "user",
            avatar="🧑"
        ):

            st.markdown(question)

        # --------------------------------------------------------------
        # Check API Key
        # --------------------------------------------------------------
        if not api_key:

            missing_key_msg = (
                "⚠️ مفتاح Groq API غير معرّف!\n\n"
                "يرجى إضافته في Streamlit Secrets "
                "باسم `GROQ_API_KEY`."
            )

            with st.chat_message(
                "assistant",
                avatar="🚑"
            ):

                st.error(
                    missing_key_msg
                )

            st.session_state.chat_history.append({

                "role": "assistant",

                "content": missing_key_msg,

                "sources": []

            })

            st.stop()

        # --------------------------------------------------------------
        # Assistant
        # --------------------------------------------------------------
        with st.chat_message(
            "assistant",
            avatar="🚑"
        ):

            with st.spinner(
                "جارِ البحث عن أفضل إجابة..."
            ):

                (
                    answer_en,
                    answer_ar,
                    sources_payload,
                    error,
                    correction_info
                ) = answer_question(

                    question,

                    index,

                    embedding_model,

                    reranker

                )

            # ----------------------------------------------------------
            # Show Query Correction
            # ----------------------------------------------------------
            if (
                correction_info
                and correction_info.get(
                    "was_corrected"
                )
            ):

                st.caption(
                    "✏️ تم تصحيح السؤال تلقائيًا إلى: "
                    f"*{correction_info['corrected']}*"
                )

            # ==========================================================
            # No Sources
            # ==========================================================
            if error == "no_sources":

                msg_text = (
                    "معنديش معلومة موثوقة عن السؤال ده "
                    "في المرجع، حاول تصيغه بشكل مختلف 🙏"
                )

                st.warning(
                    msg_text
                )

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": msg_text,

                    "sources": []

                })

            # ==========================================================
            # LLM Error
            # ==========================================================
            elif error == "llm_error":

                debug_error_msg = (
                    "⚠️ حدث خطأ أثناء الاتصال بالنموذج "
                    f"(LLM Error): {answer_en}"
                )

                st.error(
                    debug_error_msg
                )

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": debug_error_msg,

                    "sources": []

                })

            # ==========================================================
            # Success
            # ==========================================================
            else:

                st.markdown(
                    "**🇬🇧 English**"
                )

                st.markdown(
                    answer_en
                )

                st.markdown("---")

                st.markdown(
                    "**🇪🇬 بالعربي**"
                )

                st.markdown(
                    answer_ar
                )

                # ------------------------------------------------------
                # Sources
                # ------------------------------------------------------
                if (
                    sources_payload
                    and show_sources
                ):

                    st.markdown(
                        "**📚 المصادر**"
                    )

                    render_sources(
                        sources_payload
                    )

                # ------------------------------------------------------
                # Save Assistant Message
                # ------------------------------------------------------
                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content_en": answer_en,

                    "content_ar": answer_ar,

                    "sources": sources_payload,

                })

    # ==================================================================
    # Disclaimer
    # ==================================================================
    st.markdown(

        "<div class='disclaimer' "
        "style='margin-top:1.6rem;'>"
        "⚠️ هذا الموقع للأغراض التعليمية فقط "
        "ولا يغني عن استشارة طبية أو الاتصال بالطوارئ عند الحاجة."
        "</div>",

        unsafe_allow_html=True

    )


# ======================================================================
# Run App
# ======================================================================
if __name__ == "__main__":

    main()
