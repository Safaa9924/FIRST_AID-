"""
rag_core.py
------------
Shared, LIGHTWEIGHT logic used by both:
  - build_index.py  (runs ONCE, locally, to process the PDF and build the index)
  - app.py           (the Streamlit website, runs on every question)

This file intentionally does NOT import docling / pymupdf / wordfreq,
because those are only needed once while building the index, not while
the website answers questions. Keeping this file light keeps the
deployed Streamlit app fast to install and start.
"""

import os
import re
import time
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
# Values here can be overridden with environment variables or
# Streamlit "secrets" (see README.md) so you never have to hard
# code local paths / URLs into the code.

CONFIG = {
    "SOURCE_TITLE": "First Aid Reference Guide, 4th Edition — St. John Ambulance Canada",
    "PUBLICATION_YEAR": 2019,
    "IMAGES_DIR": "pdf_images",
    "DATA_DIR": "data",

    # ---------------- Retrieval ----------------
    "TOP_K": 40,
    "TOP_N_RERANK": 10,
    "BM25_WEIGHT": 0.5,
    "SEMANTIC_WEIGHT": 0.5,

    # ---------------- Context ----------------
    "MAX_CONTEXT_CHUNKS": 8,
    "WORD_BUDGET": 1500,
    "MAX_CHUNK_WORDS_IN_CONTEXT": 180,
    "MIN_CHUNK_SCORE": 1.0,

    # ---------------- LLM ----------------
    # Two backends are supported:
    #  1) Groq (cloud, works from Streamlit Community Cloud) — used
    #     automatically if GROQ_API_KEY is set.
    #  2) Ollama (local server) — used otherwise. Only reachable if the
    #     app itself is running on the same machine/network as Ollama.
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
    "GROQ_URL": "https://api.groq.com/openai/v1/chat/completions",
    "OLLAMA_URL": os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "LLM_MODEL_NAME": os.environ.get("LLM_MODEL_NAME", "llama3.2"),
    "LLM_TEMPERATURE": 0.1,
    "LLM_MAX_TOKENS": 1200,
    "LLM_SEED": 42,

    # ---------------- Correction layer ----------------
    "ENABLE_QUERY_CORRECTION": True,
    "ENABLE_CONTEXT_REUNIFICATION": True,

    # ---------------- Models ----------------
    "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
    "RERANKER_MODEL_NAME": "cross-encoder/ms-marco-MiniLM-L12-v2",
}

RESULT_COLS = ["retriever", "chunk_id", "score", "chunk_text", "page_number", "section"]


# ============================================================
# Basic helpers (identical to the notebook)
# ============================================================
def simple_tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def confidence_label(score, min_score=None):
    min_score = CONFIG["MIN_CHUNK_SCORE"] if min_score is None else min_score
    if score < min_score:
        return "Rejected"
    elif score < min_score * 1.8:
        return "Medium"
    else:
        return "High"


# ============================================================
# BM25 (must stay importable from this exact module path so that
# pickle.load() in app.py can find the class that build_index.py
# pickled).
# ============================================================
class MiniBM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.doc_lens = [len(d) for d in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 0
        self.term_freqs = [Counter(d) for d in tokenized_docs]
        self.df = Counter()
        for d in tokenized_docs:
            self.df.update(set(d))
        self.idf = {t: np.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N, dtype=np.float32)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                scores[i] += (idf * tf * (self.k1 + 1)) / denom
        return scores


# ============================================================
# Query expansion (same dictionary as the notebook)
# ============================================================
QUERY_EXPANSION = {
    "burn": ["thermal burn", "chemical burn", "critical burn", "burn dressing", "cool water"],
    "fracture": ["splint", "immobilization", "broken bone"],
    "stroke": ["FAST", "facial drooping", "speech difficulty"],
    "choking": ["back blows", "abdominal thrusts", "airway obstruction"],
    "cpr": ["chest compression", "compression rate", "cardiopulmonary resuscitation"],
    "compression": ["cpr", "chest compression rate"],
    "faint": ["fainting", "syncope", "recovery position"],
    "unconscious": ["recovery position", "breathing casualty"],
    "angina": ["heart attack", "chest pain", "cardiac"],
    "heart attack": ["angina", "chest pain", "cardiac"],
    "head injury": ["spinal injury", "skull fracture", "concussion"],
    "spinal": ["head injury", "skull fracture", "immobilization"],
    "epipen": ["anaphylaxis", "auto-injector", "allergic reaction"],
    "anaphylaxis": ["epipen", "auto-injector", "allergic reaction"],
    "naloxone": ["opioid overdose", "narcan"],
    "overdose": ["naloxone", "opioid"],
    "diabetic": ["diabetes", "hypoglycemia", "blood sugar"],
    "seizure": ["convulsion", "epilepsy"],
    "heat stroke": ["heat exhaustion", "hyperthermia"],
    "frostbite": ["hypothermia", "cold exposure"],
    "aed": ["defibrillator", "automated external defibrillator"],
    "shock": ["signs of shock", "circulatory collapse"],
    "bleeding": ["hemorrhage", "direct pressure", "wound care"],
}


def expand_query(query):
    expanded = query
    lower_query = query.lower()
    for keyword, synonyms in QUERY_EXPANSION.items():
        if keyword in lower_query:
            expanded += " " + " ".join(synonyms)
    return expanded


# ============================================================
# Unified LLM caller — talks to Groq if GROQ_API_KEY is set,
# otherwise falls back to a local Ollama server.
# Returns the raw text on success, or a string starting with
# "__LLM_ERROR__:" on failure (never raises).
# ============================================================
def _call_llm(prompt, temperature=None, max_tokens=None, seed=None):
    temperature = CONFIG["LLM_TEMPERATURE"] if temperature is None else temperature
    max_tokens = max_tokens or CONFIG["LLM_MAX_TOKENS"]
    seed = CONFIG["LLM_SEED"] if seed is None else seed

    if CONFIG["GROQ_API_KEY"]:
        # Groq's API is OpenAI-compatible: /chat/completions with a
        # "messages" list, not Ollama's "/api/generate" format.
        try:
            response = requests.post(
                CONFIG["GROQ_URL"],
                headers={
                    "Authorization": f"Bearer {CONFIG['GROQ_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": CONFIG["LLM_MODEL_NAME"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"__LLM_ERROR__:{e}"

    # Fallback: local Ollama server
    try:
        response = requests.post(
            CONFIG["OLLAMA_URL"],
            json={
                "model": CONFIG["LLM_MODEL_NAME"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature, "num_predict": max_tokens, "seed": seed,
                    "top_k": 20, "top_p": 0.8, "repeat_penalty": 1.1,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip() or "The language model returned an empty response."
    except Exception as e:
        return f"__LLM_ERROR__:{e}"


# ============================================================
# Query correction / language handling
# ============================================================

# قاموس بمصطلحات الإسعافات الأولية الشائعة بالعربي. بيستخدم في طبقة
# تصحيح حتمية (deterministic) بتتفعل للأسئلة القصيرة (كلمة أو اتنين)،
# وبتصحح بس لو التشابه قوي جدًا مع مصطلح معروف — عشان تمسك أخطاء زي
# "اغماق" -> "إغماء" من غير ما تخمّن كلمات مختلفة تمامًا زي "أعماق".
ARABIC_FIRST_AID_GLOSSARY = [
    "نزيف", "نزيف حاد", "نزيف الأنف", "كسر", "كسر مفتوح", "حروق", "حرق كهربائي",
    "حروق كيميائية", "اختناق", "انسداد مجرى التنفس", "إغماء", "فقدان الوعي",
    "الإنعاش القلبي الرئوي", "دوخة", "لسعة حشرة", "حساسية", "صدمة تحسسية",
    "صدمة كهربائية", "غرق", "تسمم", "جرعة زائدة", "سكتة دماغية", "ذبحة صدرية",
    "نوبة قلبية", "صرع", "تشنجات", "ضربة شمس", "انخفاض حرارة الجسم", "قضمة الصقيع",
    "جرح عميق", "التواء", "خلع مفصل", "إصابة الرأس", "إصابة العمود الفقري",
    "اختناق بجسم غريب", "غيبوبة سكر", "هبوط سكر الدم", "لدغة ثعبان",
]

_GLOSSARY_MATCH_THRESHOLD = 0.55


def _closest_glossary_term(text):
    best_term, best_ratio = None, 0.0
    for term in ARABIC_FIRST_AID_GLOSSARY:
        ratio = SequenceMatcher(None, text, term).ratio()
        if ratio > best_ratio:
            best_ratio, best_term = ratio, term
    return best_term, best_ratio


def correct_user_query(query, use_llm=True):
    stripped = query.strip()

    # طبقة 1 — تصحيح حتمي (مش LLM) لمصطلحات شائعة: بتتفعل بس للأسئلة
    # القصيرة جدًا (كلمة أو اتنين)، وبتصحح بس لو في تطابق قوي جدًا مع
    # مصطلح معروف. سريعة، ثابتة، ومش محتاجة اتصال إنترنت.
    if stripped and len(stripped.split()) <= 3:
        glossary_term, ratio = _closest_glossary_term(stripped)
        if glossary_term and ratio >= _GLOSSARY_MATCH_THRESHOLD and glossary_term != stripped:
            return glossary_term, True

    if not use_llm:
        return query, False

    prompt = (
        "You are a strict spelling and grammar checker. You are NOT a first-aid "
        "expert and you must NOT try to guess what the user 'probably meant'.\n\n"
        "TASK: fix ONLY obvious spelling/typing mistakes in the text below.\n\n"
        "STRICT RULES:\n"
        "1. Do NOT change the meaning or topic in any way.\n"
        "2. Do NOT invent a new question, and do NOT guess the user's intent.\n"
        "3. Do NOT answer the question or add any information to it.\n"
        "4. Keep the exact same language as the input (if it's Arabic, the output "
        "must be Arabic; if English, output must be English).\n"
        "5. If the text is unclear, nonsensical, a single ambiguous word, unrelated "
        "to first aid, or you are not fully certain what to fix, return it EXACTLY "
        "as given, character for character, with nothing changed.\n"
        "6. Return ONLY the resulting text, nothing else — no explanation, no "
        "quotation marks, no preamble.\n\n"
        f"Text: {query}"
    )
    corrected = _call_llm(prompt, temperature=0.0, max_tokens=100)
    if corrected.startswith("__LLM_ERROR__"):
        return query, False

    corrected = corrected.strip().strip('"').strip()
    if not corrected or corrected.lower() == query.lower():
        return query, False

    # حماية 1 — حارس اللغة: ارفض أي "تصحيح" غيّر أبجدية النص بالكامل
    # (يعني النموذج ترجم السؤال بدل ما يصحح إملاءه). نتحقق بعدّ نسبة
    # الحروف العربية في كل نص بدل الاعتماد على langdetect (بيبقى غير
    # موثوق مع الجمل القصيرة أو الكلمة الواحدة).
    def _arabic_ratio(text):
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
        return arabic / len(letters)

    orig_ratio = _arabic_ratio(query)
    corr_ratio = _arabic_ratio(corrected)
    # لو الأصل عربي بوضوح والناتج مبقاش عربي بوضوح (أو العكس) — ده مش
    # تصحيح إملائي، ده تغيير لغة كامل. ارفضه.
    if (orig_ratio > 0.5) != (corr_ratio > 0.5):
        return query, False

    # حماية 2 — لو الناتج مختلف جدًا عن السؤال الأصلي (يعني النموذج
    # اخترع سؤال تاني بدل ما يصحح إملاء فعليًا)، نرفض "التصحيح" ونرجّع
    # السؤال الأصلي زي ما هو بدل ما نغيّر قصد المستخدم.
    similarity = SequenceMatcher(None, query.lower(), corrected.lower()).ratio()
    if similarity < 0.45:
        return query, False

    return corrected, True


def detect_language(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def translate_to_english(text):
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def translate_to_arabic(text):
    if not text or not text.strip():
        return ""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="ar").translate(text)
    except Exception:
        return text


# ============================================================
# Retrieval (TF-IDF / BM25 / Semantic / Hybrid)
# ============================================================
def retrieve_top_k_tfidf(query, tfidf_vectorizer, tfidf_matrix, chunks_df, k=40):
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"
    return results[RESULT_COLS].reset_index(drop=True)


def retrieve_top_k_bm25(query, bm25, chunks_df, k=40):
    scores = bm25.get_scores(simple_tokenize(query))
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = np.array(scores)[ranking]
    results["retriever"] = "BM25"
    return results[RESULT_COLS].reset_index(drop=True)


def retrieve_top_k_semantic(query, embedding_model, embedding_matrix, chunks_df, k=40):
    q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(q_emb, embedding_matrix).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "Embeddings"
    return results[RESULT_COLS].reset_index(drop=True)


def retrieve_top_k_hybrid(query, bm25, embedding_model, embedding_matrix, chunks_df,
                           bm25_weight=None, semantic_weight=None, k=40):
    bm25_weight = CONFIG["BM25_WEIGHT"] if bm25_weight is None else bm25_weight
    semantic_weight = CONFIG["SEMANTIC_WEIGHT"] if semantic_weight is None else semantic_weight

    bm25_scores = min_max_normalize(bm25.get_scores(simple_tokenize(query)))

    q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    semantic_scores = min_max_normalize(cosine_similarity(q_emb, embedding_matrix).flatten())

    final_scores = bm25_weight * bm25_scores + semantic_weight * semantic_scores

    ranking = np.argsort(final_scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = final_scores[ranking]
    results["retriever"] = "Hybrid (BM25+Semantic)"
    return results[RESULT_COLS].reset_index(drop=True)


def rerank_candidates(query, candidates_df, reranker, top_n=10):
    df = candidates_df.copy().reset_index(drop=True)
    df["original_rank"] = range(1, len(df) + 1)
    pairs = [(query, text) for text in df["chunk_text"]]
    df["rerank_score"] = reranker.predict(pairs)
    df = df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    df["new_rank"] = range(1, len(df) + 1)
    return df.head(top_n)


# ============================================================
# Context building
# ============================================================
def reunify_fragmented_context(candidates_df):
    if candidates_df.empty:
        return candidates_df
    df = candidates_df.copy()
    group_order = (
        df.groupby("section")["rerank_score"].max().sort_values(ascending=False).index.tolist()
    )
    ordered_frames = []
    for section in group_order:
        grp = df[df["section"] == section].sort_values(
            by=["page_number", "chunk_id"], na_position="last"
        )
        ordered_frames.append(grp)
    return pd.concat(ordered_frames, ignore_index=True)


def build_context_package(query, reranked_df, max_context_chunks=None, word_budget=None,
                           max_chunk_words=None, min_chunk_score=None):
    max_context_chunks = max_context_chunks or CONFIG["MAX_CONTEXT_CHUNKS"]
    word_budget = word_budget or CONFIG["WORD_BUDGET"]
    max_chunk_words = max_chunk_words or CONFIG["MAX_CHUNK_WORDS_IN_CONTEXT"]
    min_chunk_score = CONFIG["MIN_CHUNK_SCORE"] if min_chunk_score is None else min_chunk_score

    candidates = reranked_df[reranked_df["rerank_score"] >= min_chunk_score].copy()
    rejected_count = len(reranked_df) - len(candidates)

    if candidates.empty:
        return {
            "query": query, "selected_df": pd.DataFrame(),
            "context_text": "", "num_sources": 0, "used_words": 0,
            "rejected_low_score": rejected_count,
        }

    if CONFIG["ENABLE_CONTEXT_REUNIFICATION"]:
        candidates = reunify_fragmented_context(candidates)
    else:
        candidates = candidates.sort_values("rerank_score", ascending=False)

    selected_rows, seen_texts, used_words = [], set(), 0

    for _, row in candidates.iterrows():
        text = row["chunk_text"].strip()
        normalized = re.sub(r"\s+", " ", text).lower()
        if normalized in seen_texts:
            continue

        words = text.split()
        if len(words) > max_chunk_words:
            text = " ".join(words[:max_chunk_words])
        chunk_words = len(text.split())

        if used_words + chunk_words > word_budget:
            break

        row = row.copy()
        row["chunk_text"] = text
        selected_rows.append(row)
        seen_texts.add(normalized)
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)

    context_blocks = []
    for i, row in selected_df.iterrows():
        page_str = f"Page {row['page_number']}" if pd.notna(row["page_number"]) and row["page_number"] != -1 else "Page N/A"
        context_blocks.append(
            f"[Source {i+1} — Section: {row['section']} — {page_str} — Score: {row['rerank_score']:.2f}]\n"
            f"{row['chunk_text']}"
        )
    context_text = ("\n\n" + "=" * 80 + "\n\n").join(context_blocks)

    return {
        "query": query, "selected_df": selected_df, "context_text": context_text,
        "num_sources": len(selected_df), "used_words": used_words,
        "rejected_low_score": rejected_count,
    }


# ============================================================
# Prompt + LLM
# ============================================================
def build_chat_prompt(condition: str, context_text: str):
    if not context_text or not context_text.strip():
        raise ValueError("Retrieved context is empty.")

    return f'''You are an expert Evidence-Based First Aid Assistant.
Answer the user question strictly in ENGLISH using ONLY the provided context. Never add outside knowledge.
If the answer is not in the context, output: "I couldn't find this information in the retrieved first aid reference."

CORE RULES:
1. Concise & Direct: Max 5 bullet points per section. No introductions or summaries.
2. No Repetition: Mention each piece of advice only once.
3. Omit Empty Sections: If a section has no context data, skip its header completely.
4. Clean Markdown: Strictly follow the structure below.

STRUCTURE TO FOLLOW:
## First Aid: {condition}

## Immediate Actions
- [Essential steps, max 5 items]

## Avoid
- [Warnings directly related, max 5 items]

## When to Call Emergency Services
- [Specific situations]

## Additional Notes
- [Extra crucial info, if any]

============================
USER QUESTION: {condition}
============================
RETRIEVED CONTEXT:
{context_text}
'''


NO_CONTEXT_EN = ("I couldn't find reliable information for this question in the retrieved "
                  "first aid reference (all matching sources scored below the confidence threshold).")
NO_CONTEXT_AR = ("لم أتمكن من العثور على معلومات موثوقة لهذا السؤال في مرجع الإسعافات الأولية "
                  "المسترجع (كل المصادر المطابقة كانت أقل من حد الثقة المطلوب).")


def generate_answer(prompt, model=None, temperature=None, max_tokens=None, seed=None):
    """Calls the configured LLM backend (Groq if GROQ_API_KEY is set,
    otherwise Ollama). Never raises — returns a string starting with
    '__LLM_ERROR__:' on failure so the Streamlit app can show a
    friendly message instead of crashing."""
    return _call_llm(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)


def evaluate_answer(answer, context_text, embedding_model):
    if not answer or not context_text:
        return None
    embeddings = embedding_model.encode([answer, context_text], convert_to_numpy=True, normalize_embeddings=True)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    if similarity >= 0.80:
        quality = "Excellent"
    elif similarity >= 0.60:
        quality = "Good"
    elif similarity >= 0.40:
        quality = "Moderate"
    else:
        quality = "Poor"
    return float(similarity), quality
