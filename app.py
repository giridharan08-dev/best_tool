# app.py
import streamlit as st
import pandas as pd
import tempfile
import backend
import chromadb

st.set_page_config(page_title="Chunking Optimizer", layout="wide")
st.markdown("""
    <style>
        h1,h2,h3,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 { color: #FF8000 !important; }
        .card { background: rgba(255,255,255,0.02); border-radius:10px; padding:18px; margin-bottom:14px; }
        .sidebar-card { background:#f7f7f7; padding:12px; border-radius:8px; border:1px solid #ddd; }
        .stButton button { background: linear-gradient(145deg,#FF8000,#FFA64D); color:white; border-radius:10px; padding:8px 18px; box-shadow:1px 1px 6px rgba(0,0,0,0.12); }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Chunking Optimizer — Sequential Flow")

STAGES = ["upload","dtype","layer1","layer2","metadata","quality","chunk","embed","store","retrieve"]
LABELS = {
    "upload":"1. Upload","dtype":"2. DTypes","layer1":"3. Preprocess L1","layer2":"4. Preprocess L2",
    "metadata":"5. Metadata","quality":"6. Quality","chunk":"7. Chunking","embed":"8. Embedding","store":"9. Store","retrieve":"10. Retrieve"
}

def render_progress(stage):
    st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.sidebar.title("Progress")
    for s in STAGES:
        if STAGES.index(s) < STAGES.index(stage):
            st.sidebar.markdown(f"✅ <span style='color:green'>{LABELS[s]}</span>", unsafe_allow_html=True)
        elif s == stage:
            st.sidebar.markdown(f"🟠 <span style='color:orange;font-weight:bold'>{LABELS[s]}</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"⚪ <span style='color:grey'>{LABELS[s]}</span>", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# session init
if "stage" not in st.session_state: st.session_state.stage = "upload"
if "df" not in st.session_state: st.session_state.df = None
if "uploaded_file_path" not in st.session_state: st.session_state.uploaded_file_path = None
if "metadata_columns" not in st.session_state: st.session_state.metadata_columns = []
if "chunks" not in st.session_state: st.session_state.chunks = None
if "metadatas" not in st.session_state: st.session_state.metadatas = None
if "collection" not in st.session_state: st.session_state.collection = None
if "model_obj" not in st.session_state: st.session_state.model_obj = None
if "embedding_model_name" not in st.session_state: st.session_state.embedding_model_name = None

def goto(s): st.session_state.stage = s
render_progress(st.session_state.stage)

# -------------------------
# UPLOAD
# -------------------------
if st.session_state.stage == "upload":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 1 — Upload CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        with st.spinner("Loading CSV..."):
            df = backend.load_csv(uploaded)
            st.session_state.df = df
            # save file for any file-based chunkers
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                st.session_state.uploaded_file_path = tmp.name
        st.success("CSV Loaded ✅")
        st.subheader("Preview (first 5 rows)")
        st.dataframe(backend.preview_data(df, 5))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed to Change DTypes"):
                goto("dtype")
        with col2:
            if st.button("Skip to Layer 1"):
                goto("layer1")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# DTYPE
# -------------------------
elif st.session_state.stage == "dtype":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 2 — Change Column DTypes (Optional)")
    df = st.session_state.df
    st.table(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))
    cols = df.columns.tolist()
    cols_to_change = st.multiselect("Select columns to change dtype", options=cols)
    for c in cols_to_change:
        choice = st.selectbox(f"Target dtype for {c}", ["Keep","str","int","float","datetime"], key=f"dtype_{c}")
    if st.button("Apply DType Changes"):
        df2 = df.copy()
        for c in cols_to_change:
            t = st.session_state.get(f"dtype_{c}", "Keep")
            if t != "Keep":
                df2, err = backend.change_dtype(df2, c, t)
                if err:
                    st.warning(f"{c} -> {t} failed: {err}")
        st.session_state.df = df2
        st.success("DType changes applied")
        st.dataframe(backend.preview_data(df2, 5))
    if st.button("Proceed to Layer 1"):
        goto("layer1")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# LAYER 1
# -------------------------
elif st.session_state.stage == "layer1":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 3 — Preprocessing Layer 1 (default actions)")
    st.write("Layer 1 will: remove HTML, strip whitespace, remove delimiters, normalize multiline cells, replace bad tokens with NaN, normalize headers.")
    if st.button("Apply Layer 1"):
        with st.spinner("Applying Layer 1..."):
            df2 = backend.layer1_preprocessing(st.session_state.df)
            st.session_state.df = df2
        st.success("Layer 1 applied ✅")
        st.write("Summary: removed html, lowercased, stripped whitespace, removed delimiters/multiline, normalized headers.")
        st.subheader("Preview (first 5 rows)")
        st.dataframe(backend.preview_data(df2, 5))
    if st.button("Proceed to Layer 2"):
        goto("layer2")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# LAYER 2
# -------------------------
elif st.session_state.stage == "layer2":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 4 — Preprocessing Layer 2 (optional transforms)")
    df = st.session_state.df
    missing_total = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    st.write(f"Missing values total: *{missing_total}* — Duplicate rows: *{dup_count}*")
    col1, col2 = st.columns(2)
    if missing_total > 0:
        fill_choice = col1.selectbox("Missing handling", ["none","fill","drop"], key="l2_missing")
        if fill_choice == "fill":
            fill_value = col1.text_input("Fill value", value="Unknown", key="l2_fill")
        else:
            fill_value = None
    else:
        fill_choice = "none"; fill_value = None
    dup_action = col2.selectbox("Duplicate handling", ["keep","drop"], index=0 if dup_count==0 else 1)
    st.markdown("Text normalization options (toggle as needed):")
    apply_stemming = st.checkbox("Stemming", key="l2_stem")
    apply_lemmatization = st.checkbox("Lemmatization", key="l2_lemma")
    remove_stopwords = st.checkbox("Remove stopwords", key="l2_stop")
    c1, c2 = st.columns(2)
    if c1.button("Apply Layer 2"):
        with st.spinner("Applying Layer 2..."):
            df2 = df.copy()
            if fill_choice == "drop":
                df2 = backend.handle_missing_values(df2, None)  # drop approach handled by user choice, but here we fill None then drop NA
                df2 = df2.dropna().reset_index(drop=True)
            elif fill_choice == "fill":
                df2 = backend.handle_missing_values(df2, fill_value)
            df2 = backend.handle_duplicates(df2, dup_action)
            df2 = backend.normalize_text(df2, apply_stemming, apply_lemmatization, remove_stopwords)
            st.session_state.df = df2
        st.success("Layer 2 applied ✅")
        st.subheader("Preview (first 5 rows)")
        st.dataframe(backend.preview_data(df2, 5))
        st.subheader("Null Summary")
        st.dataframe(backend.null_summary(df2))
        st.download_button("⬇️ Download Processed CSV", df2.to_csv(index=False).encode("utf-8"), "processed.csv")
        st.subheader("Metadata Report")
        st.json(backend.generate_metadata_report(df2))
    if c2.button("Skip and Proceed"):
        goto("metadata")
    if st.button("Proceed to Metadata selection"):
        goto("metadata")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# METADATA SELECTION
# -------------------------
elif st.session_state.stage == "metadata":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 5 — Choose Metadata Columns (for retrieval filtering)")
    df = st.session_state.df
    cols = df.columns.tolist()
    selected = st.multiselect("Select which columns should be stored as metadata for retrieval filters", options=cols, default=st.session_state.metadata_columns)
    st.session_state.metadata_columns = selected
    if st.button("Proceed to Quality Gates"):
        goto("quality")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# QUALITY GATES
# -------------------------
elif st.session_state.stage == "quality":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 6 — Quality Gates")
    df = st.session_state.df
    if df is None:
        st.warning("No data available.")
    else:
        result = backend.check_quality_gates(df)
        st.subheader("Quality summary")
        st.write(f"Rows: *{result['rows']}* — Columns: *{result['columns']}*")
        st.write(f"Total nulls: *{result['total_nulls']}* — Duplicate rows: *{result['duplicate_rows']}* ({result['duplicate_fraction']*100:.2f}% )")
        st.markdown("Null percent per column:")
        st.json(result["null_percentage_per_column"])
        if result["status"] == "PASS":
            st.success("✅ Quality Gate PASSED")
        else:
            st.error("❌ Quality Gate FAILED")
        if st.button("Proceed to Chunking"):
            goto("chunk")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# CHUNKING
# -------------------------
elif st.session_state.stage == "chunk":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 7 — Chunking (choose strategy)")
    df = st.session_state.df
    choice = st.selectbox("Chunking strategy", [
        "Fixed (row batching)",
        "Recursive (key:value + recursive splitter)",
        "Semantic+Recursive (compress→recursive)",
        "Semantic (LangChain+HF / Agentic-type)",
        "Document-based (one row = one chunk)"
    ])
    # Show parameters selectively
    if choice == "Fixed (row batching)":
        rows_per_batch = st.number_input("Rows per batch", min_value=1, max_value=1000, value=50)
    elif choice in ["Recursive (key:value + recursive splitter)","Semantic+Recursive (compress→recursive)"]:
        chunk_size = st.number_input("Chunk size (chars)", min_value=50, max_value=5000, value=400)
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=chunk_size-1, value=50)
    else:
        # semantic LangChain and document-based don't need chunk_size
        chunk_size = None; overlap = None

    if st.button("Run Chunking"):
        with st.spinner(f"Running {choice}..."):
            try:
                if choice == "Fixed (row batching)":
                    chunks = backend.fixed_row_batching(df, rows_per_batch)
                elif choice == "Recursive (key:value + recursive splitter)":
                    chunks = backend.recursive_chunk(df, chunk_size, overlap)
                elif choice == "Semantic+Recursive (compress→recursive)":
                    chunks = backend.semantic_recursive_chunk(df, chunk_size, overlap)
                elif choice == "Semantic (LangChain+HF / Agentic-type)":
                    # try langchain-based semantic chunking, fallback to cosine grouping
                    chunks = backend.semantic_chunking_langchain(df)
                else:  # Document-based
                    chunks = backend.document_based_chunking(df)

                # build metadatas using selected metadata columns (best-effort: if chunks > rows, metadata is row-level)
                metadatas = backend.build_row_metadatas(df, st.session_state.metadata_columns)
                st.session_state.chunks = chunks
                st.session_state.metadatas = metadatas
                st.success(f"Created {len(chunks)} chunks")
                for c in chunks[:3]:
                    st.code(c[:400])
            except Exception as e:
                st.error(f"Chunking failed: {e}")
    if st.session_state.chunks and st.button("Proceed to Embedding"):
        goto("embed")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# EMBEDDING
# -------------------------
elif st.session_state.stage == "embed":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 8 — Embedding")
    chunks = st.session_state.chunks
    if not chunks:
        st.warning("No chunks available. Run chunking first.")
    else:
        model_choice = st.selectbox("Embedding model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"], index=0)
        if st.button("Generate Embeddings"):
            with st.spinner("Generating embeddings..."):
                try:
                    collection, model_obj = backend.embed_and_store(chunks, model_name=model_choice, metadatas=st.session_state.metadatas, collection_name="my_collection")
                    st.session_state.collection = collection
                    st.session_state.model_obj = model_obj
                    st.session_state.embedding_model_name = model_choice
                    st.success("Embeddings created & stored in temporary collection")
                except Exception as e:
                    st.error(f"Embedding/storage failed: {e}")
    if st.button("Proceed to Store (optional)"):
        goto("store")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# STORE
# -------------------------
elif st.session_state.stage == "store":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 9 — Store (ChromaDB)")
    if st.session_state.collection is None:
        st.info("Embeddings not present yet. Run embedding first.")
    else:
        store_choice = st.radio("Persist collection to disk?", ["No", "Yes"], index=1)
        if store_choice == "Yes":
            # our embed_and_store already used PersistentClient by default; give user a confirmation
            st.success("Collection persisted (if PersistentClient available).")
    if st.button("Proceed to Retrieval"):
        goto("retrieve")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# RETRIEVAL
# -------------------------
elif st.session_state.stage == "retrieve":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 9 — Retrieval")

    q = st.text_input("Enter query")
    k = st.slider("Top-k", 1, 20, 5)

    # 🔹 Metadata Filters with numeric range support
    st.subheader("Metadata Filters")
    filters = {}
    for col in st.session_state.metadata_columns:
        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
            filter_type = st.radio(
                f"Filter for {col}",
                ["Skip", "Exact", "Greater than", "Less than", "Between"],
                key=f"ft_{col}"
            )
            if filter_type == "Exact":
                val = st.number_input(f"Enter exact value for {col}", key=f"{col}_exact")
                filters[col] = val
            elif filter_type == "Greater than":
                val = st.number_input(f"Enter min value for {col}", key=f"{col}_gt")
                filters[col] = {"gt": val}
            elif filter_type == "Less than":
                val = st.number_input(f"Enter max value for {col}", key=f"{col}_lt")
                filters[col] = {"lt": val}
            elif filter_type == "Between":
                min_val = st.number_input(f"Min {col}", key=f"{col}_min")
                max_val = st.number_input(f"Max {col}", key=f"{col}_max")
                filters[col] = {"min": min_val, "max": max_val}
        else:
            val = st.text_input(f"Enter exact match for {col}", key=f"{col}_str")
            if val:
                filters[col] = val

    # 🔎 Run search
    if st.button("Search"):
        with st.spinner("Searching..."):
            res = backend.search_query(
                st.session_state.collection,
                st.session_state.model_obj,
                q, k, filters=filters
            )
            docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            with st.expander(f"Result Rank {i+1} (distance {dist:.4f})"):
                st.markdown(f"<p style='line-height:1.6'>{doc}</p>", unsafe_allow_html=True)
                if meta:
                    st.json(meta)
                else:
                    st.caption("No metadata available")

    st.markdown('</div>', unsafe_allow_html=True)