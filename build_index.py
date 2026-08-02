"""
build_index.py
----------------
Run this ONCE, locally, BEFORE deploying the Streamlit app.

It takes the First Aid PDF, processes it exactly like the original
notebook (clean -> chunk -> TF-IDF -> BM25 -> embeddings -> extract
images), and saves everything the website needs into:

    data/chunks_df.pkl
    data/tfidf_index.pkl
    data/bm25_index.pkl
    data/embedding_matrix.npy
    data/page_to_images.pkl
    pdf_images/*.png (or whatever format the PDF used)

Usage:
    python build_index.py --pdf "First aid reference guide_V4.1_Public.pdf"

This step needs heavier libraries (docling, pymupdf, wordfreq) that
the deployed website itself does NOT need — that's why it's a
separate script from app.py.
"""

import os
import re
import argparse
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from rag_core import CONFIG, MiniBM25, simple_tokenize


# ============================================================
# Stage 1: PDF loading (docling)
# ============================================================
def load_pdf_document(pdf_path):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.generate_picture_images = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(pdf_path)
    doc = result.document

    page_items = []
    text_parts = []

    for item, _ in doc.iterate_items():
        if not hasattr(item, "text"):
            continue
        text = item.text.strip()
        if not text:
            continue

        item_type = item.__class__.__name__

        page_no = None
        prov = getattr(item, "prov", None)
        if prov:
            try:
                page_no = int(prov[0].page_no)
            except Exception:
                page_no = None

        if item_type == "SectionHeaderItem":
            rendered = f"\n## {text}\n"
        elif item_type == "ListItem":
            rendered = f"- {text}"
        else:
            rendered = text

        text_parts.append(rendered)
        page_items.append({"text": rendered, "item_type": item_type, "page_no": page_no})

    raw_text = "\n\n".join(text_parts)
    return {
        "source_file": os.path.basename(pdf_path),
        "raw_text": raw_text,
        "word_count": len(raw_text.split()),
        "page_items": page_items,
    }


# ============================================================
# Stage 1b: Extract images per page (PyMuPDF)
# ============================================================
def extract_pdf_images_by_page(pdf_path, output_dir):
    import fitz  # PyMuPDF

    os.makedirs(output_dir, exist_ok=True)
    src = fitz.open(pdf_path)
    page_to_images = defaultdict(list)

    for page_number, page in enumerate(src, start=1):
        images = page.get_images(full=True)
        for local_idx, img in enumerate(images, start=1):
            xref = img[0]
            try:
                extracted = src.extract_image(xref)
                image_bytes = extracted["image"]
                ext = extracted["ext"]
                image_name = f"page_{page_number:04d}_{local_idx}.{ext}"
                image_path = os.path.join(output_dir, image_name)
                with open(image_path, "wb") as fh:
                    fh.write(image_bytes)
                page_to_images[page_number].append(image_path)
            except Exception as e:
                print("Skipped image", xref, "page", page_number, e)

    return dict(page_to_images)


# ============================================================
# Stage 2: Cleaning
# ============================================================
CUSTOM_VOCAB = {
    "defibrillator", "defibrillators", "cardiopulmonary", "epinephrine",
    "tourniquet", "anaphylaxis", "hypothermia",
}


def is_word(w, lang="en"):
    from wordfreq import zipf_frequency
    w = w.lower()
    if w in CUSTOM_VOCAB:
        return True
    return zipf_frequency(w, lang) > 0


def fix_ligature_breaks(text):
    from wordfreq import zipf_frequency
    FRAG_ZIPF_MAX = 5.0
    COMBINED_ZIPF_MIN = 2.0

    tokens = list(re.finditer(r"[A-Za-z]+", text))
    out = []
    last_end = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        merged = False
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            gap = text[tok.end():nxt.start()]
            frag, rest = tok.group(), nxt.group()
            if gap == " " and 1 <= len(frag) <= 6:
                combined = frag + rest
                frag_zipf = zipf_frequency(frag.lower(), "en")
                combined_zipf = zipf_frequency(combined.lower(), "en")
                if combined_zipf >= COMBINED_ZIPF_MIN and frag_zipf < FRAG_ZIPF_MAX:
                    out.append(text[last_end:tok.start()])
                    out.append(combined)
                    last_end = nxt.end()
                    i += 2
                    merged = True
        if not merged:
            i += 1
    out.append(text[last_end:])
    return "".join(out)


def fix_hyphen_linebreaks(text):
    pattern = re.compile(r"(\w+)-\s*\n\s*(\w+)")

    def repl(m):
        w1, w2 = m.group(1), m.group(2)
        no_hyphen = w1 + w2
        with_hyphen = w1 + "-" + w2
        return no_hyphen if is_word(no_hyphen) else with_hyphen

    return pattern.sub(repl, text)


def fix_pdf_broken_words(text):
    text = fix_hyphen_linebreaks(text)
    text = fix_ligature_breaks(text)
    return text


def prepare_document(page_items, headings, front_matter_anchor=r"Chapter\s+1\s+Introduction\s+to\s+First\s+Aid"):
    headings_sorted = sorted([h.strip() for h in headings if h and h.strip()], key=len, reverse=True)

    start_idx = 0
    anchor_re = re.compile(front_matter_anchor, re.IGNORECASE)
    for i, item in enumerate(page_items):
        if anchor_re.search(item["text"]):
            start_idx = i
            break

    items = page_items[start_idx:]
    cleaned_items = []
    previous_line = ""

    for item in items:
        text = item["text"]
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = fix_pdf_broken_words(text)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"(?m)^-(?=[A-Za-z])", "- ", text)

        for heading in headings_sorted:
            pattern = r"(?<!\n)" + re.escape(" " + heading)
            text = re.sub(pattern, f"\n\n## {heading}", text)

        text = re.sub(r"[ \t]+", " ", text).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text or text == previous_line:
            continue

        cleaned_items.append({"text": text, "item_type": item["item_type"], "page_no": item["page_no"]})
        previous_line = text

    cleaned_text = "\n\n".join(it["text"] for it in cleaned_items)
    return cleaned_items, cleaned_text


# ============================================================
# Stage 3-4: Chunking
# ============================================================
def compute_document_statistics(items):
    word_counts = [len(it["text"].split()) for it in items if it["text"].strip()]
    if not word_counts:
        return {}, []
    arr = np.array(word_counts)
    return {
        "p25_words_per_item": round(float(np.percentile(arr, 25)), 1),
        "median_words_per_item": round(float(np.median(arr)), 1),
        "p90_words_per_item": round(float(np.percentile(arr, 90)), 1),
    }, word_counts


def semantic_chunk_with_pages(items, target_words, overlap_words, min_chunk_words, max_chunk_words):
    sections = []
    current_heading = "General"
    current_paragraphs = []

    for it in items:
        text, page_no = it["text"], it["page_no"]
        if text.startswith("#"):
            if current_paragraphs:
                sections.append((current_heading, current_paragraphs))
            current_heading = text
            current_paragraphs = []
        else:
            current_paragraphs.append((text, page_no))

    if current_paragraphs:
        sections.append((current_heading, current_paragraphs))

    chunks = []
    for heading, paragraphs in sections:
        buffer_words, current_pages = [heading.split()], set()
        current_words = len(heading.split())

        def flush():
            nonlocal buffer_words, current_words, current_pages
            words_flat = [w for grp in buffer_words for w in grp]
            if words_flat:
                text_out = " ".join(words_flat)
                pages_sorted = sorted(p for p in current_pages if p is not None)
                chunks.append({
                    "chunk_text": text_out,
                    "word_count": len(words_flat),
                    "page_number": pages_sorted[0] if pages_sorted else None,
                    "page_range": pages_sorted,
                })
            buffer_words, current_words, current_pages = [heading.split()], len(heading.split()), set()

        for para_text, page_no in paragraphs:
            para_words = para_text.split()
            if current_words + len(para_words) > max_chunk_words and current_words >= min_chunk_words:
                flush()
            buffer_words.append(para_words)
            current_words += len(para_words)
            if page_no is not None:
                current_pages.add(page_no)
            if current_words >= target_words:
                flush()

        if current_words > len(heading.split()):
            flush()

    merged = []
    carry = None
    for ch in chunks:
        if carry is not None:
            ch["chunk_text"] = carry["chunk_text"] + " " + ch["chunk_text"]
            ch["word_count"] = carry["word_count"] + ch["word_count"]
            ch["page_range"] = sorted(set(carry["page_range"]) | set(ch["page_range"]))
            ch["page_number"] = ch["page_range"][0] if ch["page_range"] else carry["page_number"]
            carry = None
        if ch["word_count"] < min_chunk_words:
            carry = ch
            continue
        merged.append(ch)
    if carry is not None:
        merged.append(carry)

    return merged


def process_chunk_metadata(chunks_df, source_title, publication_year, pdf_path, words_per_minute=200):
    df = chunks_df.copy()
    df["source_title"] = source_title
    df["publication_year"] = int(publication_year) if publication_year else None
    df["source_file"] = os.path.basename(pdf_path)

    header_extract = df["chunk_text"].str.extract(r"^(?P<h_level>#+)\s*(?P<section>.+)$", expand=True)
    df["section"] = header_extract["section"].fillna("General").str.strip()
    df["header_level"] = header_extract["h_level"].str.len().fillna(0).astype(int)

    df["avg_word_length"] = (df["char_count"] / df["word_count"].replace(0, np.nan)).round(2).fillna(0.0)
    df["read_time_sec"] = (df["word_count"] / (words_per_minute / 60)).round(1)
    return df


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the First Aid PDF")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    print("Stage 1: loading PDF with docling ...")
    pdf_document = load_pdf_document(pdf_path)

    print("Stage 1b: extracting images per page ...")
    page_to_images = extract_pdf_images_by_page(pdf_path, CONFIG["IMAGES_DIR"])

    print("Stage 2: cleaning text ...")
    headings = [it["text"].lstrip("#").strip() for it in pdf_document["page_items"] if it["item_type"] == "SectionHeaderItem"]
    cleaned_items, _ = prepare_document(pdf_document["page_items"], headings)

    print("Stage 3: computing chunk-size bounds from document statistics ...")
    stats, _ = compute_document_statistics(cleaned_items)
    min_chunk_words = int(np.clip(stats["p25_words_per_item"], 40, 120))
    target_chunk_words = int(np.clip(stats["median_words_per_item"] * 3, 150, 350))
    max_chunk_words = int(np.clip(stats["p90_words_per_item"] * 3, 250, 500))
    overlap_words = int(target_chunk_words * 0.30)

    print("Stage 4: semantic chunking ...")
    raw_chunks = semantic_chunk_with_pages(
        cleaned_items, target_words=target_chunk_words, overlap_words=overlap_words,
        min_chunk_words=min_chunk_words, max_chunk_words=max_chunk_words,
    )
    chunks_df = pd.DataFrame(raw_chunks)
    chunks_df.insert(0, "chunk_id", [f"chunk_{i:04d}" for i in range(1, len(chunks_df) + 1)])
    chunks_df["char_count"] = chunks_df["chunk_text"].apply(len)

    print("Stage 6: chunk metadata ...")
    chunks_df = process_chunk_metadata(chunks_df, CONFIG["SOURCE_TITLE"], CONFIG["PUBLICATION_YEAR"], pdf_path)
    print(f"Generated {len(chunks_df)} chunks.")

    print("Stage 7a: TF-IDF index ...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_df=0.90, max_features=30000,
        sublinear_tf=True, norm="l2", dtype="float32",
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks_df["chunk_text"].tolist())

    print("Stage 7b: BM25 index ...")
    tokenized_docs = [simple_tokenize(t) for t in chunks_df["chunk_text"]]
    bm25 = MiniBM25(tokenized_docs)

    print("Stage 8: semantic embeddings (downloads the model on first run) ...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(CONFIG["EMBEDDING_MODEL_NAME"])
    embedding_matrix = embedding_model.encode(
        chunks_df["chunk_text"].tolist(), batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print("Saving everything to data/ ...")
    os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
    with open(os.path.join(CONFIG["DATA_DIR"], "tfidf_index.pkl"), "wb") as f:
        pickle.dump({"vectorizer": tfidf_vectorizer, "matrix": tfidf_matrix}, f)
    with open(os.path.join(CONFIG["DATA_DIR"], "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    np.save(os.path.join(CONFIG["DATA_DIR"], "embedding_matrix.npy"), embedding_matrix)
    with open(os.path.join(CONFIG["DATA_DIR"], "page_to_images.pkl"), "wb") as f:
        pickle.dump(page_to_images, f)
    chunks_df.to_pickle(os.path.join(CONFIG["DATA_DIR"], "chunks_df.pkl"))

    print("=" * 60)
    print("DONE. Files created in data/:", os.listdir(CONFIG["DATA_DIR"]))
    print(f"Images extracted for {len(page_to_images)} pages into '{CONFIG['IMAGES_DIR']}/'")
    print("Now upload the 'data/' and 'pdf_images/' folders together with the rest of the app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
