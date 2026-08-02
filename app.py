"""
app.py — First Aid RAG website (Streamlit)

Before running this app you MUST run build_index.py once locally to
generate the data/ folder and pdf_images/ folder. This app only
loads the already-built index — it does not process the PDF itself.
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

st.set_page_config(page_title="First Aid Assistant | مساعد الإسعافات الأولية", page_icon="🩹", layout="centered")

DATA_DIR = CONFIG["DATA_DIR"]
IMAGES_DIR = CONFIG["IMAGES_DIR"]


# ============================================================
# Load the pre-built index (cached so it only loads once)
# ============================================================
@st.cache_resource(show_spinner="Loading knowledge base ...")
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


@st.cache_resource(show_spinner="Loading AI models (first run only, may take a minute) ...")
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedding_model = SentenceTransformer(CONFIG["EMBEDDING_MODEL_NAME"])
    reranker = CrossEncoder(CONFIG["RERANKER_MODEL_NAME"])
    return embedding_model, reranker


# ============================================================
# UI
# ============================================================
st.title("🩹 First Aid Assistant")
st.caption("مساعد الإسعافات الأولية — اسأل بالعربي أو بالإنجليزي")

index = load_index()

if index is None:
    st.error(
        "لم يتم العثور على ملفات الفهرس داخل مجلد `data/`.\n\n"
        "لازم تشغّل `build_index.py` مرة واحدة محليًا الأول عشان يعالج ملف الـ PDF "
        "وينتج الملفات المطلوبة، بعدين ترفع مجلد `data/` (ومجلد `pdf_images/` لو "
        "عايز الصور تظهر) مع باقي ملفات المشروع.\n\n"
        "راجع ملف README.md لمعرفة الخطوات بالتفصيل."
    )
    st.stop()

with st.spinner("جاري تجهيز النموذج ..."):
    embedding_model, reranker = load_models()

with st.sidebar:
    st.header("⚙️ الإعدادات")
    st.write(f"عدد المقاطع في القاعدة المعرفية: **{len(index['chunks_df'])}**")
    use_llm_correction = st.checkbox("تصحيح السؤال تلقائيًا (يحتاج LLM)", value=False)
    show_sources = st.checkbox("إظهار المصادر والصفحات", value=True)
    st.markdown("---")
    st.caption(
        "ملاحظة: توليد الإجابة النهائية يحتاج نموذج لغوي (LLM) متصل عبر "
        "`OLLAMA_URL`. لو مش متاح، هيظهر لك السياق المسترجَع من الكتاب فقط "
        "بدون صياغة نهائية بالذكاء الاصطناعي."
    )

user_question = st.text_input("اكتب سؤالك عن الإسعافات الأولية:", placeholder="مثال: إزاي أتصرف مع حرق من الدرجة الأولى؟")
ask = st.button("اسأل", type="primary")

if ask and user_question.strip():
    question = user_question.strip()

    with st.spinner("جاري البحث والتحليل ..."):
        # 1) optional spelling correction
        if use_llm_correction and CONFIG["ENABLE_QUERY_CORRECTION"]:
            question, was_corrected = correct_user_query(question, use_llm=True)
        else:
            was_corrected = False

        # 2) language + translation to English for retrieval
        language = detect_language(question)
        retrieval_query = translate_to_english(question)
        expanded_query = expand_query(retrieval_query)

        # 3) hybrid retrieval
        results = retrieve_top_k_hybrid(
            expanded_query, index["bm25"], embedding_model, index["embedding_matrix"],
            index["chunks_df"], k=CONFIG["TOP_K"],
        )

        # 4) cross-encoder rerank
        reranked = rerank_candidates(expanded_query, results, reranker, top_n=CONFIG["TOP_N_RERANK"])

        # 5) build context
        context = build_context_package(query=expanded_query, reranked_df=reranked)

        # 6) generate answer
        if context["num_sources"] == 0:
            english_answer = None
            arabic_answer = "لم أتمكن من العثور على معلومات موثوقة لهذا السؤال في مرجع الإسعافات الأولية المسترجع."
        else:
            prompt = build_chat_prompt(condition=question, context_text=context["context_text"])
            raw = generate_answer(prompt)

            if raw and raw.startswith("__LLM_ERROR__"):
                english_answer = None
                arabic_answer = None
                llm_error = raw.split(":", 1)[1] if ":" in raw else "unknown error"
            else:
                english_answer = raw
                arabic_answer = translate_to_arabic(raw) if raw else None
                llm_error = None

    if was_corrected:
        st.info(f"تم تصحيح السؤال تلقائيًا إلى: *{question}*")

    if context["num_sources"] == 0:
        st.warning(arabic_answer)
    elif english_answer is None:
        st.error(
            "تعذّر الوصول إلى النموذج اللغوي (LLM) لتوليد إجابة نهائية "
            f"({llm_error}).\n\n"
            "هنعرض لك أقرب المقاطع اللي القاه في المرجع بدل الإجابة المُولَّدة:"
        )
        st.markdown(context["context_text"])
    else:
        tab_ar, tab_en = st.tabs(["🇸🇦 العربية", "🇬🇧 English"])
        with tab_ar:
            st.markdown(arabic_answer or "—")
        with tab_en:
            st.markdown(english_answer)

    if show_sources and context["num_sources"] > 0:
        st.markdown("---")
        st.subheader("📚 المصادر")
        sources_df = context["selected_df"][["chunk_id", "section", "page_number", "rerank_score"]].copy()
        sources_df.insert(0, "الترتيب", range(1, len(sources_df) + 1))
        sources_df["الثقة"] = sources_df["rerank_score"].apply(confidence_label)
        sources_df = sources_df.rename(columns={
            "chunk_id": "المقطع", "section": "القسم", "page_number": "الصفحة", "rerank_score": "الدرجة",
        })
        st.dataframe(sources_df, hide_index=True, use_container_width=True)

        shown_pages = set()
        for _, src_row in context["selected_df"].iterrows():
            page = src_row["page_number"]
            if pd.isna(page) or page == -1 or page in shown_pages:
                continue
            shown_pages.add(page)
            for img_path in index["page_to_images"].get(int(page), [])[:1]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"صفحة {int(page)}", width=300)

elif ask:
    st.warning("اكتب سؤالك الأول.")

st.markdown("---")
st.caption(
    "⚠️ هذا الموقع للأغراض التعليمية فقط ولا يغني عن استشارة طبية أو الاتصال بالطوارئ عند الحاجة."
)
