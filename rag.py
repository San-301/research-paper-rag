import os
import re

import fitz
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Add GEMINI_API_KEY "
            "to Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path):
    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    return embeddings


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, index, chunks, top_k=5):
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    retrieved_chunks = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):
        if index_position == -1:
            continue

        chunk = chunks[index_position].copy()
        chunk["distance"] = float(distance)

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, retrieved_chunks):
    client = get_gemini_client()

    context = build_context(retrieved_chunks)

    prompt = f"""
You are a Research Paper Question Answering Assistant.

Answer the user's question using ONLY the provided research-paper
context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "The answer is not available in the provided research paper."
3. Give a clear and concise answer.
4. Cite the relevant page number(s).
5. Do not invent information.
6. If multiple pages support the answer, cite all relevant pages.

Research Paper Context:
-----------------------
{context}
-----------------------

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def process_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "Could not create text chunks from the PDF."
        )

    embeddings = create_embeddings(chunks)

    index = create_vector_store(embeddings)

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index
    }


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question, rag_data):
    retrieved_chunks = retrieve_chunks(
        question,
        rag_data["index"],
        rag_data["chunks"],
        top_k=5
    )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer, retrieved_chunks
