# backend.py
import io
import csv
import re
import typing
import math
import warnings
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# Embeddings / models
from sentence_transformers import SentenceTransformer

# Chroma
import chromadb

# sklearn for cosine similarity fallback
from sklearn.metrics.pairwise import cosine_similarity

# Try to import RecursiveCharacterTextSplitter robustly
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None
        warnings.warn("RecursiveCharacterTextSplitter not available; recursive splitting will fallback to fixed-size splitting.")

# Try to import SemanticChunker / HuggingFaceEmbeddings (LangChain community / huggingface wrappers)
SemanticChunker = None
HuggingFaceEmbeddings = None
try:
    # preferred community embedding wrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
    SemanticChunker = SemanticChunker
except Exception:
    try:
        # alternative package
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker
        SemanticChunker = SemanticChunker
    except Exception:
        HuggingFaceEmbeddings = None
        SemanticChunker = None
        warnings.warn(
            "SemanticChunker / HuggingFaceEmbeddings not available. "
            "Install 'langchain_community' or 'langchain-huggingface' & 'langchain-experimental' for LangChain+HF chunking."
        )

# Optional CSV loader from LangChain (not required)
try:
    from langchain_community.document_loaders.csv_loader import CSVLoader
except Exception:
    CSVLoader = None

# ----------------------------
# CSV Loading / Utilities
# ----------------------------
try:
    import chardet
except Exception:
    chardet = None

def _detect_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        return ","

def load_csv(file_or_path: typing.Union[str, io.BytesIO, io.StringIO]):
    """Load CSV from path or stream file (Streamlit uploaded file)"""
    if isinstance(file_or_path, str):
        return pd.read_csv(file_or_path)

    if hasattr(file_or_path, "read"):
        file_or_path.seek(0)
        raw = file_or_path.read()

        if isinstance(raw, (bytes, bytearray)):
            encoding = "utf-8"
            if chardet:
                try:
                    res = chardet.detect(raw)
                    encoding = res.get("encoding") or "utf-8"
                except Exception:
                    pass
            text = raw.decode(encoding, errors="replace")
            sep = _detect_sep(text)
            return pd.read_csv(io.StringIO(text), sep=sep)

        if isinstance(raw, str):
            sep = _detect_sep(raw)
            return pd.read_csv(io.StringIO(raw), sep=sep)

    raise ValueError("Unsupported input for load_csv")

def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    n = min(len(df), n)
    return df.head(n).reset_index(drop=True)

# ----------------------------
# Layer 1 preprocessing helpers
# ----------------------------
def remove_html(df: pd.DataFrame) -> pd.DataFrame:
    clean_re = re.compile(r"<.*?>")
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).map(lambda s: re.sub(clean_re, " ", s))
    return df2

def to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).map(lambda s: s.lower())
    return df2

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = df2[c].astype(str).map(lambda s: s.strip())
    return df2

def remove_delimiters(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    pattern = re.compile(r"[;|,*]")
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).map(lambda s: pattern.sub(" ", s))
    return df2

def remove_multiline(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).map(lambda s: s.replace("\n", " ").replace("\r", " "))
    return df2

def clean_unrelated_values(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].map(lambda x: None if str(x).strip() in ["", "-", "*"] else x)
    return df2

def layer1_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Layer1: HTML removal, lowercasing, strip whitespace, remove delimiters, multiline, replace bad tokens, normalize headers."""
    df2 = df.copy()
    df2 = remove_html(df2)
    df2 = to_lowercase(df2)
    df2 = remove_delimiters(df2)
    df2 = remove_multiline(df2)
    df2 = clean_unrelated_values(df2)
    # normalize headers: empty -> empty_column_i
    new_cols = []
    for i, c in enumerate(df2.columns):
        name = str(c).strip()
        new_cols.append(name if name not in ["", "-", "*"] else f"empty_column_{i+1}")
    df2.columns = new_cols
    return df2

# ----------------------------
# Layer2 preprocessing helpers
# ----------------------------
def handle_missing_values(df: pd.DataFrame, fill_value: typing.Any):
    return df.fillna(fill_value)

def handle_duplicates(df: pd.DataFrame, action: str):
    if action == "drop":
        return df.drop_duplicates().reset_index(drop=True)
    return df

def normalize_text(df: pd.DataFrame, apply_stemming: bool=False, apply_lemmatization: bool=False, remove_stopwords: bool=False) -> pd.DataFrame:
    """
    Lightweight normalization using NLTK (porter stem + wordnet lemmatizer).
    If NLTK is missing, returns df unchanged.
    """
    try:
        import nltk
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.corpus import stopwords
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
    except Exception:
        return df

    ps = PorterStemmer()
    lm = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        out = []
        for val in df2[c].astype(str):
            tokens = re.split(r"\s+", val.strip())
            if remove_stopwords:
                tokens = [t for t in tokens if t.lower() not in stop_words]
            if apply_stemming:
                tokens = [ps.stem(t) for t in tokens]
            if apply_lemmatization:
                tokens = [lm.lemmatize(t) for t in tokens]
            out.append(" ".join(tokens))
        df2[c] = out
    return df2

# ----------------------------
# Null summary
# ----------------------------
def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        pct = round((null_count / total_rows * 100) if total_rows > 0 else 0, 2)
        rows.append({"column": col, "null_count": null_count, "null_percentage": pct})
    return pd.DataFrame(rows)

# ----------------------------
# Metadata report (text / numeric)
# ----------------------------
def generate_metadata_report(df: pd.DataFrame) -> Dict[str, Any]:
    report = {}
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["text_columns"] = {c: {"rows": int(df[c].count())} for c in text_cols}
    numeric_info = {}
    for c in num_cols:
        col = df[c]
        numeric_info[c] = {
            "count": int(col.count()),
            "min": float(col.min()) if not col.isnull().all() else None,
            "max": float(col.max()) if not col.isnull().all() else None,
            "mean": float(col.mean()) if not col.isnull().all() else None,
            "median": float(col.median()) if not col.isnull().all() else None,
            "nulls": int(col.isnull().sum())
        }
    report["numeric_columns"] = numeric_info
    report["processed_at"] = datetime.utcnow().isoformat() + "Z"
    return report

# ----------------------------
# Quality gates
# ----------------------------
def check_quality_gates(df: pd.DataFrame, null_threshold: float = 0.2, dup_threshold: float = 0.05) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    null_counts = df.isnull().sum()
    null_perc = (null_counts / max(1, n_rows)).fillna(0).to_dict()
    total_nulls = int(null_counts.sum())
    dup_count = int(df.duplicated().sum())
    dup_frac = dup_count / max(1, n_rows)
    fail_null = any([v > null_threshold for v in null_perc.values()])
    fail_dup = dup_frac > dup_threshold
    passed = not (fail_null or fail_dup)
    return {
        "rows": n_rows,
        "columns": n_cols,
        "total_nulls": total_nulls,
        "null_percentage_per_column": null_perc,
        "duplicate_rows": dup_count,
        "duplicate_fraction": dup_frac,
        "status": "PASS" if passed else "FAIL"
    }

# ----------------------------
# Chunking helpers
# ----------------------------
def fixed_size_chunking_from_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), step) if text[i:i+chunk_size]]
    return chunks

def fixed_row_batching(df: pd.DataFrame, rows_per_batch: int = 50) -> List[str]:
    rows_text = df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    batches = []
    for i in range(0, len(rows_text), rows_per_batch):
        batches.append("\n".join(rows_text[i:i+rows_per_batch]))
    return batches

def recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    # convert each row to key:value (compact) then recursive split
    texts = df.astype(str).apply(lambda r: ", ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    big_text = "\n".join(texts)
    if RecursiveCharacterTextSplitter is None:
        return fixed_size_chunking_from_text(big_text, chunk_size=chunk_size, overlap=overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(big_text)

def semantic_recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    # compress rows into natural language sentences then recursive split
    compressed = []
    for _, row in df.iterrows():
        pieces = [f"{c}:{str(row[c])}" for c in df.columns]
        compressed.append("; ".join(pieces))
    text = "\n".join(compressed)
    if RecursiveCharacterTextSplitter is None:
        return fixed_size_chunking_from_text(text, chunk_size=chunk_size, overlap=overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def document_based_chunking(df: pd.DataFrame) -> List[str]:
    # each row becomes its own chunk (one document per row)
    return df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()

# ----------------------------
# Semantic chunking using LangChain+HuggingFace (if available) or fallback to cosine grouping
# ----------------------------
def semantic_chunking_langchain(df: pd.DataFrame, batch_size: int = 50) -> List[str]:
    if SemanticChunker is None or HuggingFaceEmbeddings is None:
        # fallback
        return semantic_chunking_cosine(df)
    texts = df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    if not texts:
        return []
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    # batch rows
    batched = []
    for i in range(0, len(texts), batch_size):
        batched.append("\n".join(texts[i:i+batch_size]))
    docs = semantic_chunker.create_documents(batched)
    return [d.page_content for d in docs]

def semantic_chunking_cosine(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.7) -> List[str]:
    docs = df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    if not docs:
        return []
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=False)
    chunks, current_chunk, current_embs = [], [], []
    for text, emb in zip(docs, embeddings):
        if not current_chunk:
            current_chunk = [text]; current_embs = [emb]; continue
        avg_emb = np.mean(current_embs, axis=0).reshape(1, -1)
        sim = cosine_similarity(avg_emb, emb.reshape(1, -1))[0][0]
        if sim >= threshold:
            current_chunk.append(text); current_embs.append(emb)
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [text]; current_embs = [emb]
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def agentic_ai_chunking(df: pd.DataFrame) -> List[str]:
    # Use SemanticChunker on rows directly if available, else fallback to cosine grouping
    if SemanticChunker is None or HuggingFaceEmbeddings is None:
        return semantic_chunking_cosine(df)
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    texts = df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    docs = semantic_chunker.create_documents(texts)
    return [d.page_content for d in docs]

# ----------------------------
# Metadata builder for storing in Chroma
# ----------------------------
def build_row_metadatas(df: pd.DataFrame, metadata_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if metadata_columns:
        out = []
        for _, row in df.iterrows():
            md = {}
            for c in metadata_columns:
                if c in df.columns:
                    v = row.get(c)
                    if pd.isna(v):
                        md[c] = None
                    elif isinstance(v, (str, int, float, bool)):
                        md[c] = v
                    else:
                        md[c] = str(v)
            out.append(md)
        return out
    else:
        return df.to_dict(orient="records")

# ----------------------------
# Embedding + Chroma storage
# ----------------------------
def embed_and_store(chunks: List[str],
                    embeddings: Optional[List[List[float]]] = None,
                    model_name: str = "all-MiniLM-L6-v2",
                    chroma_path: str = "chromadb_store",
                    collection_name: str = "temp_collection",
                    metadatas: Optional[List[dict]] = None):
    # compute embeddings if not provided
    model = SentenceTransformer(model_name)
    if embeddings is None:
        embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
    emb_lists = [list(map(float, e)) for e in embeddings]

    # persistent client preferred
    try:
        client = chromadb.PersistentClient(path=chroma_path)
    except Exception:
        client = chromadb.Client()

    # create/get collection safely
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    # clear existing docs in collection
    try:
        existing = collection.get()
        if existing and isinstance(existing, dict) and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [str(i) for i in range(len(chunks))]

    if metadatas and len(metadatas) == len(chunks):
        # sanitize metadata
        sanitized = []
        for m in metadatas:
            dd = {}
            for k, v in (m or {}).items():
                if v is None:
                    dd[k] = None
                elif isinstance(v, (str, int, float, bool)):
                    dd[k] = v
                else:
                    dd[k] = str(v)
            sanitized.append(dd)
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists, metadatas=sanitized)
    else:
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists)

    return collection, model

# ----------------------------
# Search / retrieval with optional metadata filters
# ----------------------------
def search_query(collection, model, query: str, k: int = 5, filters: dict = None):
    q_emb = model.encode([query])
    res = collection.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    combined = [{"doc": d, "meta": m, "dist": float(dist)} for d, m, dist in zip(docs, metas, dists)]

    if filters:
        filtered = []
        for item in combined:
            meta = item["meta"]
            if not meta:
                continue
            ok = True
            for kf, vf in filters.items():
                if kf not in meta:
                    ok = False
                    break

                try:
                    meta_val = float(meta[kf])
                except:
                    meta_val = meta[kf]

                # handle numeric ranges and comparisons
                if isinstance(vf, dict):  
                    if "min" in vf and meta_val < vf["min"]:
                        ok = False; break
                    if "max" in vf and meta_val > vf["max"]:
                        ok = False; break
                    if "gt" in vf and meta_val <= vf["gt"]:
                        ok = False; break
                    if "lt" in vf and meta_val >= vf["lt"]:
                        ok = False; break
                else:
                    # exact match (string or number)
                    if str(meta_val).lower() != str(vf).lower():
                        ok = False; break
            if ok:
                filtered.append(item)
        combined = filtered

    return {
        "documents": [[c["doc"] for c in combined]],
        "metadatas": [[c["meta"] for c in combined]],
        "distances": [[c["dist"] for c in combined]],
    }