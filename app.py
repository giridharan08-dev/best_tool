# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import os
import math
import logging
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# ---------- Apply Orange-Grey Theme ----------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #FF8C00;  /* Dark Orange */
        --secondary: #FFA500; /* Orange */
        --accent: #FFB74D;    /* Light Orange */
        --dark: #2C3E50;      /* Dark Grey */
        --medium: #34495E;    /* Medium Grey */
        --light: #ECF0F1;     /* Light Grey */
        --text: #2C3E50;      /* Dark Text */
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #ECF0F1 0%, #FFFFFF 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark) !important;
        border-left: 4px solid var(--primary) !important;
        padding-left: 10px !important;
    }
    
    /* Buttons - Orange theme with hover */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, var(--secondary), var(--accent)) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--dark) 0%, var(--medium) 100%) !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--dark) 0%, var(--medium) 100%) !important;
    }
    
    /* Sidebar text */
    .css-1d391kg h1, 
    .css-1d391kg h2, 
    .css-1d391kg h3,
    .css-1d391kg p,
    .css-1d391kg label {
        color: var(--light) !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed var(--primary) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        background: rgba(255, 140, 0, 0.05) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(45deg, #d4edda, #c3e6cb) !important;
        border-left: 4px solid #28a745 !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(45deg, #f8d7da, #f5c6cb) !important;
        border-left: 4px solid #dc3545 !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(45deg, #fff3cd, #ffeaa7) !important;
        border-left: 4px solid var(--primary) !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid var(--medium) !important;
        border-radius: 8px !important;
    }
    
    .dataframe thead th {
        background: var(--primary) !important;
        color: white !important;
    }
    
    /* Hover effects for dataframe rows */
    .dataframe tbody tr:hover {
        background-color: rgba(255, 140, 0, 0.1) !important;
        transition: background-color 0.2s ease;
    }
    
    /* Columns spacing */
    .stColumn {
        padding: 10px;
    }
    
    /* Text input focus */
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border: 1px solid var(--medium) !important;
        border-radius: 4px !important;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border: 1px solid var(--medium) !important;
        border-radius: 4px !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: var(--text) !important;
        font-weight: 500 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border: 1px solid var(--medium) !important;
        border-radius: 8px !important;
        background: var(--light) !important;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        margin: 20px 0;
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        border: 1px solid var(--light);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border-left: 4px solid var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# ---------- defensive imports ----------
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENT_TRANS = True
except Exception:
    HAS_SENT_TRANS = False

try:
    import chromadb
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_RECURSIVE = True
except Exception:
    HAS_RECURSIVE = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_COSINE = True
except Exception:
    HAS_COSINE = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

# ---------- NLTK imports for text processing ----------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chunking_app")

# ---------- Streamlit page ----------
st.set_page_config(
    page_title="Chunking Optimizer", 
    layout="wide",
    page_icon="📦"
)

# Custom header with orange theme
st.markdown("""
<div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">📦 Chunking Optimizer</h1>
    <p style="color: white; text-align: center; margin: 10px 0 0 0; font-size: 1.2em;">Advanced Text Chunking and Semantic Search</p>
</div>
""", unsafe_allow_html=True)

# ---------- session defaults ----------
DEFAULTS = {
    "mode": None,
    "df": None,
    "uploaded_filename": None,
    "upload_path": None,
    "chunks": [],
    "metas": [],
    "embeddings": None,  # numpy array (N x D)
    "model": None,       # SentenceTransformer model (or None)
    "store": None,       # dict describing where stored
    "timings": {},       # dict step -> seconds
    "status": {},        # step -> pending/running/done
    "file_meta": {}
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- helper utilities ----------
def set_status(step: str, val: str, elapsed: Optional[float] = None):
    st.session_state["status"][step] = val
    if elapsed is not None:
        st.session_state["timings"][step] = elapsed
    # trigger rerun small: not strictly necessary
    st.experimental_rerun()

def log_step(step_name: str, func, *args, **kwargs):
    """Runs step and updates status + timing. returns func result or raises."""
    try:
        st.session_state["status"][step_name] = "running"
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = round(time.time() - start, 3)
        st.session_state["status"][step_name] = "done"
        st.session_state["timings"][step_name] = elapsed
        logger.info(f"Step {step_name} done in {elapsed}s")
        return res
    except Exception as e:
        st.session_state["status"][step_name] = "pending"
        logger.exception(f"Error in step {step_name}: {e}")
        raise

def pretty_kb(bytesize: Optional[int]) -> str:
    if bytesize is None: return "N/A"
    if bytesize < 1024: return f"{bytesize} B"
    kb = bytesize / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024
    return f"{mb:.2f} MB"

# ---------- simple file loader ----------
def load_csv_from_upload(file_obj) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    """Load CSV uploaded file-like into DataFrame and return minimal metadata."""
    # file_obj is a Streamlit UploadedFile
    file_obj.seek(0)
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        text = file_obj.read().decode("utf-8", errors="replace")
        from io import StringIO
        df = pd.read_csv(StringIO(text))
    meta = {
        "filename": getattr(file_obj, "name", "uploaded.csv"),
        "file_size_bytes": getattr(file_obj, "size", None),
        "loaded_at": datetime.utcnow().isoformat()
    }
    return df, meta

# ---------- advanced text preprocessing functions ----------
def remove_html_values_df(df: pd.DataFrame) -> pd.DataFrame:
    import re
    df2 = df.copy()
    pattern = re.compile(r"<.*?>")
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).map(lambda s: pattern.sub(" ", s))
    return df2

def normalize_whitespace_and_lower(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).map(lambda s: " ".join(str(s).split()).strip().lower())
    return df2

def remove_stopwords_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stopwords from text columns using NLTK"""
    if not HAS_NLTK:
        st.warning("NLTK not available. Skipping stopwords removal.")
        return df
    
    df2 = df.copy()
    stop_words = set(stopwords.words('english'))
    
    def remove_stopwords_text(text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).apply(remove_stopwords_text)
    return df2

def apply_stemming_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stemming to text columns using NLTK"""
    if not HAS_NLTK:
        st.warning("NLTK not available. Skipping stemming.")
        return df
    
    df2 = df.copy()
    stemmer = PorterStemmer()
    
    def stem_text(text):
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).apply(stem_text)
    return df2

def apply_lemmatization_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lemmatization to text columns using NLTK"""
    if not HAS_NLTK:
        st.warning("NLTK not available. Skipping lemmatization.")
        return df
    
    df2 = df.copy()
    lemmatizer = WordNetLemmatizer()
    
    def lemmatize_text(text):
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).apply(lemmatize_text)
    return df2

# ---------- key-value natural sentence generator ----------
def row_to_kv_sentence(row: pd.Series) -> str:
    # produce: "Matchid 21, where age is 25, where place is chennai"
    items = [(str(k).strip(), str(v).strip()) for k, v in row.items() if pd.notna(v) and str(v).strip() != ""]
    if not items:
        return ""
    first_k, first_v = items[0]
    sentence = f"{first_k} {first_v}"
    for k, v in items[1:]:
        sentence += f", where {k} is {v}"
    return sentence

# ---------- recursive (key-value) chunking ----------
def recursive_kv_chunking(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Convert rows to natural KV sentences, then recursively split them (langchain splitter if available)."""
    compressed = [row_to_kv_sentence(row) for _, row in df.iterrows()]
    text = "\n".join([s for s in compressed if s])
    if text == "":
        return []
    if HAS_RECURSIVE:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        parts = splitter.split_text(text)
        return parts
    # fallback: sliding by characters
    step = max(1, chunk_size - overlap)
    parts = [text[i:i+chunk_size] for i in range(0, len(text), step)]
    return parts

# ---------- semantic clustering chunking ----------
def semantic_clustering_chunking(df: pd.DataFrame, n_clusters: Optional[int] = None, chunk_size: int = 400, overlap: int = 50) -> Tuple[List[str], List[dict]]:
    """
    Semantic chunking by clustering rows into groups using embeddings (if available).
    Returns (chunks, metadatas) where metadatas correspond to chunks (group info).
    - n_clusters: optional override number of clusters; if None we compute heuristic.
    """
    # 1) build per-row short text (keys+values compact)
    row_texts = df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    if len(row_texts) == 0:
        return [], []

    # 2) compute row embeddings (use sentence-transformers if available)
    if HAS_SENT_TRANS:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        row_embs = model.encode(row_texts, show_progress_bar=False)
    else:
        # fallback deterministic pseudo-embeddings
        rng = np.random.RandomState(1234)
        dim = 256
        row_embs = rng.rand(len(row_texts), dim)

    # 3) choose number of clusters
    if n_clusters is None:
        # heuristic: one cluster per ~50 rows (bounded)
        heur = max(1, min(20, int(len(row_texts) / 50) + 1))
        n_clusters = heur

    # 4) cluster via KMeans (if available) else group by first categorical column
    if HAS_SKLEARN:
        kmeans = KMeans(n_clusters=min(n_clusters, len(row_texts)), random_state=42)
        labels = kmeans.fit_predict(row_embs)
    else:
        # fallback: single cluster
        labels = np.zeros(len(row_texts), dtype=int)

    # 5) for each cluster, join its rows into a single block and then recursively chunk that block
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(row_texts[i])

    all_chunks = []
    all_metas = []
    for lab, rows_in in clusters.items():
        block_text = "\n".join(rows_in)
        # split block_text recursively
        if HAS_RECURSIVE:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            parts = splitter.split_text(block_text)
        else:
            step = max(1, chunk_size - overlap)
            parts = [block_text[i:i+chunk_size] for i in range(0, len(block_text), step)]
        for j, p in enumerate(parts):
            all_chunks.append(p)
            all_metas.append({"cluster": int(lab), "cluster_size": len(rows_in), "chunk_index_in_cluster": j})
    return all_chunks, all_metas
# ---------- fixed row batching ----------
def fixed_row_batching(df: pd.DataFrame, rows_per_batch: int = 50) -> Tuple[List[str], List[dict]]:
    rows_text = df.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1).tolist()
    chunks = []
    metas = []
    for i in range(0, len(rows_text), rows_per_batch):
        batch = "\n".join(rows_text[i:i+rows_per_batch])
        chunks.append(batch)
        metas.append({"rows_in_batch": min(rows_per_batch, max(0, len(rows_text)-i)), "batch_index": i//rows_per_batch})
    return chunks, metas

# ---------- embedding helpers ----------
def embed_texts(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Optional[SentenceTransformer], np.ndarray]:
    if len(chunks) == 0:
        return None, np.zeros((0, 0), dtype=np.float32)
    if HAS_SENT_TRANS:
        model = SentenceTransformer(model_name)
        embs = model.encode(chunks, show_progress_bar=False)
        arr = np.asarray(embs, dtype=np.float32)
        return model, arr
    # fallback deterministic random embeddings
    rng = np.random.RandomState(1234)
    dim = 256
    arr = rng.rand(len(chunks), dim).astype(np.float32)
    return None, arr

def ensure_python_floats(emb_arr: np.ndarray) -> List[List[float]]:
    """Convert numpy float32 array to list of lists of native Python floats (safe for chroma)."""
    if emb_arr is None:
        return []
    if isinstance(emb_arr, np.ndarray):
        return emb_arr.astype(float).tolist()
    # if already list-of-lists, ensure cast
    return [[float(x) for x in vec] for vec in emb_arr]

# ---------- storage helpers ----------
def store_in_chroma(chunks: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None, collection_name: str = "chunks_collection"):
    if not HAS_CHROMA:
        return {"stored": False, "type": "memory", "chunks_count": len(chunks)}
    emb_py = ensure_python_floats(embeddings)
    try:
        try:
            client = chromadb.PersistentClient(path="chromadb_store")
        except Exception:
            client = chromadb.Client()
        # create safe collection
        try:
            col = client.get_collection(collection_name)
            # delete existing (safe)
            existing = col.get()
            if existing and existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            col = client.create_collection(collection_name)
        ids = [str(i) for i in range(len(chunks))]
        if metadatas and len(metadatas) == len(chunks):
            col.add(ids=ids, documents=chunks, embeddings=emb_py, metadatas=metadatas)
        else:
            col.add(ids=ids, documents=chunks, embeddings=emb_py)
        return {"stored": True, "type": "chroma", "collection": collection_name, "n_vectors": len(chunks)}
    except Exception as e:
        logger.exception("Chroma store failed: %s", e)
        return {"stored": False, "type": "memory", "chunks_count": len(chunks)}

def store_in_faiss(embeddings: np.ndarray):
    if not HAS_FAISS:
        return {"stored": False, "type": "memory", "n_vectors": embeddings.shape[0] if hasattr(embeddings, "shape") else 0}
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)  # expects float32
    return {"stored": True, "type": "faiss", "index": index, "n_vectors": embeddings.shape[0]}

# ---------- retrieval helpers ----------
def retrieve_from_store(query: str, model, store_info: dict, chunks: List[str], embeddings: np.ndarray, k: int = 5):
    # prepare query embedding
    if HAS_SENT_TRANS and model is not None:
        q_emb = model.encode([query])
        q_arr = np.asarray(q_emb, dtype=np.float32)
    else:
        rng = np.random.RandomState(abs(hash(query)) % (2**32))
        q_arr = rng.rand(1, embeddings.shape[1] if embeddings is not None and embeddings.shape[1]>0 else 256).astype(np.float32)

    # chroma
    if store_info and store_info.get("type") == "chroma" and HAS_CHROMA:
        try:
            try:
                client = chromadb.PersistentClient(path="chromadb_store")
            except Exception:
                client = chromadb.Client()
            col = client.get_collection(store_info["collection"])
            res = col.query(query_embeddings=q_arr.tolist(), n_results=k, include=["documents", "distances", "metadatas"])
            docs = res.get("documents", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0] if "metadatas" in res else [None]*len(docs)
            return docs, dists, metas
        except Exception as e:
            logger.exception("Chroma query failed: %s", e)
            # fallback to memory below

    # faiss
    if store_info and store_info.get("type") == "faiss" and HAS_FAISS:
        idx = store_info["index"]
        D, I = idx.search(q_arr, k)
        docs = [chunks[i] if i < len(chunks) else "" for i in I[0]]
        dists = D[0].tolist()
        return docs, dists, [None]*len(docs)

    # memory fallback (cosine)
    if embeddings is not None and embeddings.shape[0] > 0:
        if HAS_COSINE:
            sims = cosine_similarity(q_arr, embeddings)[0]  # higher => closer
            idxs = np.argsort(-sims)[:k]
            docs = [chunks[i] for i in idxs]
            dists = (1 - sims[idxs]).tolist()
            return docs, dists, [None]*len(docs)
        else:
            sims = (embeddings @ q_arr.T).reshape(-1)
            idxs = np.argsort(-sims)[:k]
            docs = [chunks[i] for i in idxs]
            return docs, [float(sims[i]) for i in idxs], [None]*len(docs)

    return [], [], []

# ---------- UI: sidebar process tracker + session summary ----------
def status_emoji(status: str) -> str:
    return {"pending":"⚪","running":"🟠","done":"✅"}.get(status, "⚪")

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0;">Process Tracker</h2>
    </div>
    """, unsafe_allow_html=True)
    
    steps = ["Upload", "Preprocessing", "Chunking", "Embedding", "Storing", "Retrieval"]

    for s in steps:
        timing = st.session_state["timings"].get(s)
        timing_str = f"({timing}s)" if timing else ""
        status = st.session_state["status"].get(s, "pending")
        st.markdown(f"""
        <div style="background: {'#FFA500' if status == 'running' else '#34495E' if status == 'done' else '#2C3E50'}; 
                    padding: 10px; border-radius: 5px; margin: 5px 0; color: white;">
            {status_emoji(status)}  <strong>{s}</strong> {timing_str}
        </div>
        """, unsafe_allow_html=True)

    if st.session_state["timings"]:
        total = sum(st.session_state["timings"].values())
        st.markdown(f"""
        <div style="background: #FF8C00; padding: 15px; border-radius: 8px; margin: 10px 0; color: white; text-align: center;">
            <h3 style="margin: 0;">⏱ Total: {total:.2f}s</h3>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: white; text-align: center; margin: 0;">Session Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"- **Mode:** {st.session_state['mode'] or 'None'}")
    st.write(f"- **File:** {st.session_state['uploaded_filename'] or 'N/A'}")
    st.write(f"- **Upload path:** {st.session_state['upload_path'] or 'N/A'}")
    fm = st.session_state.get("file_meta", {})
    st.write(f"- **Size:** {pretty_kb(fm.get('file_size_bytes'))}")
    st.write(f"- **Uploaded at:** {fm.get('loaded_at','N/A')}")
    if st.session_state["timings"]:
        total = sum(st.session_state["timings"].values())
        st.write(f"- **Total processing time:** {total:.2f}s")
    
    st.markdown("---")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        for k in DEFAULTS.keys():
            st.session_state[k] = DEFAULTS[k]
        st.experimental_rerun()
# ---------- main layout ----------
st.markdown("## 🎯 Choose a Mode")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("⚡ Fast Mode — Auto optimized", use_container_width=True):
        st.session_state["mode"] = "fast"
with c2:
    if st.button("⚙️ Config-1 — High-level options", use_container_width=True):
        st.session_state["mode"] = "config1"
with c3:
    if st.button("🔬 Deep Config — Advanced tuning", use_container_width=True):
        st.session_state["mode"] = "deep"

st.markdown(f"**Selected mode:** `{st.session_state['mode']}`")

# ---------- show upload area (only after selecting mode) ----------
if st.session_state["mode"]:
    st.markdown("### 📤 Upload CSV")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        df, meta = load_csv_from_upload(uploaded)
        st.session_state["df"] = df
        st.session_state["uploaded_filename"] = meta["filename"]
        # save temp copy in uploads folder for trace
        fname = f"{int(time.time())}_{meta['filename']}"
        save_path = os.path.join(UPLOAD_DIR, fname)
        uploaded.seek(0)
        with open(save_path, "wb") as fh:
            fh.write(uploaded.read())
        st.session_state["upload_path"] = save_path
        st.session_state["file_meta"] = {"file_size_bytes": meta.get("file_size_bytes"), "loaded_at": meta.get("loaded_at")}
        
        st.success(f"✅ Loaded {meta['filename']} — rows: {len(df)}, cols: {len(df.columns)}")
        st.markdown(f"""
        <div style="background: #ECF0F1; padding: 15px; border-radius: 8px; border-left: 4px solid #FF8C00;">
            <p><strong>📊 Dataset Summary:</strong></p>
            <p>• Rows: {len(df)}</p>
            <p>• Columns: {len(df.columns)}</p>
            <p>• Size: {pretty_kb(meta.get('file_size_bytes'))}</p>
        </div>
        """, unsafe_allow_html=True)
        # set Upload done
        st.session_state["status"]["Upload"] = "done"

    # ---------- FAST MODE UI ----------
    if st.session_state["mode"] == "fast":
        st.markdown("### ⚡ Fast Mode")
        st.markdown("**Auto semantic clustering + chunking + embed + store**")
        st.write("Default pipeline: Auto-preprocess → semantic-clustering chunking → embed → store (chroma).")
        
        if st.button("▶ Run Fast Pipeline", type="primary"):
            if st.session_state["df"] is None:
                st.error("📝 Please upload a CSV file first.")
            else:
                # Preprocessing (auto)
                df0 = st.session_state["df"]
                df1 = log_step("Preprocessing", lambda d: normalize_whitespace_and_lower(remove_html_values_df(d)), df0)
                st.session_state["df"] = df1

                # Chunking: semantic clustering default
                chunks, metas = log_step("Chunking", semantic_clustering_chunking, df1, None, 400, 50)
                st.session_state["chunks"] = chunks
                st.session_state["metas"] = metas

                # Embedding
                model, emb_arr = log_step("Embedding", embed_texts, chunks, "all-MiniLM-L6-v2")
                st.session_state["model"] = model
                st.session_state["embeddings"] = emb_arr

                # Storing: chroma preferred, fallback to memory
                store_info = log_step("Storing", store_in_chroma, chunks, emb_arr, metas, f"fast_collection_{int(time.time())}")
                st.session_state["store"] = store_info

                st.success("✅ Fast pipeline completed successfully!")
                st.markdown(f"""
                <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <p><strong>📈 Pipeline Results:</strong></p>
                    <p>• Chunks created: {len(chunks)}</p>
                    <p>• Storage: {store_info.get('type', 'memory')}</p>
                    <p>• Status: {'✅ Stored' if store_info.get('stored') else '⚠️ Memory only'}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state["status"]["Retrieval"] = "pending"

    # ---------- CONFIG-1 UI ----------
    elif st.session_state["mode"] == "config1":
        st.markdown("### ⚙️ Config-1 Mode")
        st.markdown("**Preprocessing → Chunking → Embedding → Storing**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Preprocessing Options")
            null_handling = st.selectbox("Missing values", ["keep", "drop", "fill_with_value"])
            fill_value = None
            if null_handling == "fill_with_value":
                fill_value = st.text_input("Fill value", "Unknown")
            
            st.markdown("#### 📦 Chunking Options")
            chunk_method = st.selectbox("Chunk method", ["semantic_cluster", "recursive_kv", "fixed_row", "document_row"])
            chunk_size = st.number_input("Chunk size (chars)", 100, 5000, 400, step=50)
            overlap = st.number_input("Overlap (chars)", 0, 2000, 50, step=10)
        
        with col2:
            st.markdown("#### 🤖 Embedding & Storage")
            embed_choice = st.selectbox("Embedding model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"])
            storage_choice = st.selectbox("Vector storage", ["chroma", "faiss", "memory"])
            
            st.markdown("#### ⚡ Quick Actions")
            if st.button("▶ Run Config-1 Pipeline", type="primary"):
                if st.session_state["df"] is None:
                    st.error("📝 Please upload a CSV file first.")
                else:
                    df0 = st.session_state["df"].copy()
                    # Preprocessing
                    if null_handling == "drop":
                        df1 = log_step("Preprocessing", lambda d: d.dropna().reset_index(drop=True), df0)
                    elif null_handling == "fill_with_value":
                        df1 = log_step("Preprocessing", lambda d, v: d.fillna(v), df0, fill_value)
                    else:
                        df1 = log_step("Preprocessing", normalize_whitespace_and_lower, df0)
                    st.session_state["df"] = df1

                    # Chunking
                    if chunk_method == "semantic_cluster":
                        chunks, metas = log_step("Chunking", semantic_clustering_chunking, df1, None, chunk_size, overlap)
                    elif chunk_method == "recursive_kv":
                        chunks = log_step("Chunking", recursive_kv_chunking, df1, chunk_size, overlap)
                        metas = [{} for _ in chunks]
                    elif chunk_method == "fixed_row":
                        chunks, metas = log_step("Chunking", fixed_row_batching, df1, 50)
                    else:  # document_row
                        chunks = df1.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df1.columns]), axis=1).tolist()
                        metas = [{} for _ in chunks]

                    st.session_state["chunks"] = chunks
                    st.session_state["metas"] = metas

                    # Embedding
                    model, emb_arr = log_step("Embedding", embed_texts, chunks, embed_choice)
                    st.session_state["model"] = model
                    st.session_state["embeddings"] = emb_arr

                    # Storing
                    if storage_choice == "chroma":
                        store_info = log_step("Storing", store_in_chroma, chunks, emb_arr, metas, f"config1_{int(time.time())}")
                    elif storage_choice == "faiss":
                        store_info = log_step("Storing", store_in_faiss, emb_arr)
                    else:
                        store_info = {"stored": False, "type": "memory", "n_vectors": emb_arr.shape[0] if emb_arr is not None else 0}
                    st.session_state["store"] = store_info
                    
                    st.success("✅ Config-1 pipeline completed successfully!")
                    st.markdown(f"""
                    <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                        <p><strong>📈 Pipeline Results:</strong></p>
                        <p>• Chunks created: {len(chunks)}</p>
                        <p>• Chunking method: {chunk_method}</p>
                        <p>• Storage: {store_info.get('type', 'memory')}</p>
                        <p>• Status: {'✅ Stored' if store_info.get('stored') else '⚠️ Memory only'}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ---------- DEEP CONFIG UI (FIXED) ----------
    elif st.session_state["mode"] == "deep":
        st.markdown("### 🔬 Deep Config Mode")
        st.markdown("**Advanced preprocessing and chunking options**")
        
        st.markdown("#### 🛠️ Advanced Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🧹 Data Cleaning")
            null_handling = st.selectbox("Null value handling", ["keep", "drop", "fill_with_value"], key="deep_null")
            fill_val = None
            if null_handling == "fill_with_value":
                fill_val = st.text_input("Custom fill value", "Unknown", key="deep_fill")
            
            duplicate_action = st.selectbox("Duplicate rows", ["keep", "drop"], key="deep_dup")
            
            st.markdown("##### 📝 Text Preprocessing")
            rm_html = st.checkbox("Remove HTML tags", value=True, key="deep_html")
            norm_ws = st.checkbox("Normalize whitespace", value=True, key="deep_ws")
            to_lower = st.checkbox("Convert text to lowercase", value=True, key="deep_lower")
        
        with col2:
            st.markdown("##### 🧠 NLP Processing")
            remove_stop = st.checkbox("Remove stopwords", key="deep_stop")
            do_stem = st.checkbox("Apply stemming", key="deep_stem")
            do_lemma = st.checkbox("Apply lemmatization", key="deep_lemma")
            
            st.markdown("##### 📦 Chunking Options")
            chunk_method = st.selectbox("Chunk method", 
                                       ["semantic_cluster", "recursive_kv", "fixed_row", "document_row"],
                                       key="deep_chunk_method")
            chunk_size = st.number_input("Chunk size (chars)", 100, 5000, 400, step=50, key="deep_chunk_size")
            overlap = st.number_input("Overlap (chars)", 0, 2000, 50, step=10, key="deep_overlap")
            
            st.markdown("##### 💾 Storage Options")
            storage_choice = st.selectbox("Vector storage", ["chroma", "faiss", "memory"], key="deep_storage")

        # Additional options for fixed row batching
        if chunk_method == "fixed_row":
            rows_per_batch = st.number_input("Rows per batch", 1, 100, 10, key="deep_rows_batch")

        if st.button("▶ Run Deep Config Pipeline", type="primary"):
            if st.session_state["df"] is None:
                st.error("📝 Please upload a CSV file first.")
            else:
                df0 = st.session_state["df"].copy()

                # Null handling
                if null_handling == "drop":
                    df0 = log_step("Preprocessing - Drop Nulls", lambda d: d.dropna().reset_index(drop=True), df0)
                elif null_handling == "fill_with_value":
                    df0 = log_step("Preprocessing - Fill Nulls", lambda d, val: d.fillna(val), df0, fill_val)

                # Duplicate handling
                if duplicate_action == "drop":
                    df0 = log_step("Preprocessing - Drop Duplicates", lambda d: d.drop_duplicates().reset_index(drop=True), df0)

                # Text preprocessing in sequence
                if rm_html:
                    df0 = log_step("Preprocessing - Remove HTML", remove_html_values_df, df0)
                if norm_ws:
                    df0 = log_step("Preprocessing - Normalize WS", normalize_whitespace_and_lower, df0)
                if to_lower:
                    df0 = log_step("Preprocessing - Lowercase", 
                                  lambda d: d.applymap(lambda x: str(x).lower() if isinstance(x, str) else x), 
                                  df0)
                if remove_stop:
                    df0 = log_step("Preprocessing - Remove Stopwords", remove_stopwords_df, df0)
                if do_stem:
                    df0 = log_step("Preprocessing - Stemming", apply_stemming_df, df0)
                if do_lemma:
                    df0 = log_step("Preprocessing - Lemmatization", apply_lemmatization_df, df0)

                st.session_state["df"] = df0

                # Chunking based on selected method
                if chunk_method == "semantic_cluster":
                    chunks, metas = log_step("Chunking", semantic_clustering_chunking, df0, None, chunk_size, overlap)
                elif chunk_method == "recursive_kv":
                    chunks = log_step("Chunking", recursive_kv_chunking, df0, chunk_size, overlap)
                    metas = [{} for _ in chunks]
                elif chunk_method == "fixed_row":
                    chunks, metas = log_step("Chunking", fixed_row_batching, df0, rows_per_batch)
                else:  # document_row
                    chunks = df0.astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df0.columns]), axis=1).tolist()
                    metas = [{} for _ in chunks]

                st.session_state["chunks"] = chunks
                st.session_state["metas"] = metas

                # Embedding
                model, emb_arr = log_step("Embedding", embed_texts, chunks, "all-MiniLM-L6-v2")
                st.session_state["model"] = model
                st.session_state["embeddings"] = emb_arr

                # Storing
                if storage_choice == "chroma":
                    store_info = log_step("Storing", store_in_chroma, chunks, emb_arr, metas, f"deep_config_{int(time.time())}")
                elif storage_choice == "faiss":
                    store_info = log_step("Storing", store_in_faiss, emb_arr)
                else:
                    store_info = {"stored": False, "type": "memory", "n_vectors": emb_arr.shape[0] if emb_arr is not None else 0}
                
                st.session_state["store"] = store_info
                
                st.success("✅ Deep config pipeline completed successfully!")
                st.markdown(f"""
                <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <p><strong>📈 Pipeline Results:</strong></p>
                    <p>• Chunks created: {len(chunks)}</p>
                    <p>• Chunking method: {chunk_method}</p>
                    <p>• Preprocessing steps: {sum([rm_html, norm_ws, to_lower, remove_stop, do_stem, do_lemma])}</p>
                    <p>• Storage: {store_info.get('type', 'memory')}</p>
                    <p>• Status: {'✅ Stored' if store_info.get('stored') else '⚠️ Memory only'}</p>
                </div>
                """, unsafe_allow_html=True)

# ---------- Retrieval / test UI ----------
if st.session_state["store"] is not None and (st.session_state["embeddings"] is not None):
    st.markdown("---")
    st.markdown("### 🔍 Retrieval / Test")
    st.markdown("Test your chunking and embedding results with natural language queries.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        q = st.text_input("Enter query to test retrieval (natural language)", placeholder="Search for similar content...")
    with col2:
        k = st.slider("Top K results", min_value=1, max_value=10, value=3)
    
    if q:
        docs, dists, metas = log_step("Retrieval", retrieve_from_store, q, st.session_state["model"], st.session_state["store"], st.session_state["chunks"], st.session_state["embeddings"], k)
        if docs:
            st.success(f"✅ Found {len(docs)} results!")
            st.session_state["status"]["Retrieval"] = "done"
            
            for i, (doc, dist) in enumerate(zip(docs, dists)):
                similarity_score = 1 - dist if dist <= 1 else 1/(1+dist)  # Normalize score
                score_color = "#28a745" if similarity_score > 0.7 else "#ffc107" if similarity_score > 0.4 else "#dc3545"
                
                st.markdown(f"""
                <div style="background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {score_color};">
                    <h4 style="margin: 0 0 10px 0; color: {score_color};">
                        Result {i+1} (Score: {similarity_score:.3f})
                    </h4>
                    <p style="margin: 0; color: #666; font-size: 0.9em;">{doc[:500]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                if metas and i < len(metas) and metas[i]:
                    with st.expander(f"View Metadata for Result {i+1}"):
                        st.json(metas[i])
        else:
            st.warning("No results found for your query.")

# ---------- Save / Export area ----------
if st.session_state["chunks"]:
    st.markdown("---")
    st.markdown("### 💾 Exports")
    st.markdown("Download your processed chunks and embeddings for external use.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.download_button("📥 Save Chunks (.txt)", "\n\n---\n\n".join(st.session_state["chunks"]), file_name="chunks.txt", use_container_width=True):
            st.success("Chunks download prepared")
    
    with col2:
        if st.session_state["embeddings"] is not None:
            buf = io.BytesIO()
            np.save(buf, st.session_state["embeddings"])
            buf.seek(0)
            if st.download_button("📥 Save Embeddings (.npy)", buf, file_name="embeddings.npy", use_container_width=True):
                st.success("Embeddings download prepared")
# ---------- final session summary ----------
st.markdown("---")
st.markdown("### 📊 Session Summary")
st.markdown("Quick overview of your current session.")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("""
    <div style="background: #ECF0F1; padding: 15px; border-radius: 8px; text-align: center;">
        <h4 style="color: #FF8C00; margin: 0;">📁 File Info</h4>
        <p style="margin: 5px 0;"><strong>File:</strong> {}</p>
        <p style="margin: 5px 0;"><strong>Size:</strong> {}</p>
    </div>
    """.format(
        st.session_state['uploaded_filename'] or 'N/A',
        pretty_kb(st.session_state.get('file_meta', {}).get('file_size_bytes'))
    ), unsafe_allow_html=True)

with summary_col2:
    st.markdown("""
    <div style="background: #ECF0F1; padding: 15px; border-radius: 8px; text-align: center;">
        <h4 style="color: #FF8C00; margin: 0;">⚙️ Processing</h4>
        <p style="margin: 5px 0;"><strong>Mode:</strong> {}</p>
        <p style="margin: 5px 0;"><strong>Chunks:</strong> {}</p>
    </div>
    """.format(
        st.session_state['mode'] or 'None',
        len(st.session_state['chunks']) if st.session_state['chunks'] else 0
    ), unsafe_allow_html=True)

with summary_col3:
    if st.session_state['timings']:
        total = sum(st.session_state['timings'].values())
        st.markdown("""
        <div style="background: #ECF0F1; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #FF8C00; margin: 0;">⏱ Timings</h4>
            <p style="margin: 5px 0;"><strong>Total:</strong> {:.2f}s</p>
            <p style="margin: 5px 0;"><strong>Steps:</strong> {}</p>
        </div>
        """.format(total, len(st.session_state['timings'])), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;">
    <p>📦 Chunking Optimizer • Built with Streamlit • Orange-Grey Theme</p>
</div>
""", unsafe_allow_html=True)                        