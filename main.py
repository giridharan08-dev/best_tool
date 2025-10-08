# main.py - COMPLETE UPDATED VERSION WITH FIXED DATABASE IMPORT
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import io
import numpy as np
import tempfile
import os
from typing import Optional
import json
import uvicorn
import shutil
from backend import (
    run_fast_pipeline, 
    run_config1_pipeline, 
    run_deep_pipeline, 
    retrieve_similar,
    export_chunks,
    export_embeddings,
    get_system_info,
    get_file_info,
    connect_mysql,
    connect_postgresql,
    get_table_list,
    import_table_to_dataframe,
    process_large_file,
    can_load_file,
    LARGE_FILE_THRESHOLD,
    process_file_direct,
    EMBEDDING_BATCH_SIZE,
    PARALLEL_WORKERS,
    export_preprocessed_data  # NEW: Added export function
)

app = FastAPI(title="Chunking Optimizer API", version="2.0")

# ---------------------------
# OpenAI-compatible API Endpoints
# ---------------------------
@app.post("/v1/embeddings")
async def openai_embeddings(
    model: str = Form("text-embedding-ada-002"),
    input: str = Form(...),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible embeddings endpoint"""
    try:
        from backend import OpenAIEmbeddingAPI
        
        embedding_api = OpenAIEmbeddingAPI(
            model_name=model,
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        # Handle both string and list of strings
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
            
        embeddings = embedding_api.encode(texts)
        
        # Format response in OpenAI standard
        response_data = {
            "object": "list",
            "data": [],
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        }
        
        for i, embedding in enumerate(embeddings):
            response_data["data"].append({
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i
            })
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.post("/v1/chat/completions")
async def openai_chat_completions(
    model: str = Form("gpt-3.5-turbo"),
    messages: str = Form(...),
    max_tokens: Optional[int] = Form(1000),
    temperature: Optional[float] = Form(0.7),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible chat completions endpoint (requires external OpenAI API)"""
    try:
        import openai
        
        if openai_api_key:
            openai.api_key = openai_api_key
        if openai_base_url:
            openai.base_url = openai_base_url
            
        # Parse messages from JSON string
        messages_list = json.loads(messages)
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return JSONResponse(content=response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")

# ---------------------------
# Enhanced DB IMPORT with Large File Support (FIXED)
# ---------------------------
@app.post("/db/test_connection")
async def db_test_connection(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"status": "error", "message": "Unsupported db_type"}
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/db/list_tables")
async def db_list_tables(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        tables = get_table_list(conn, db_type)
        conn.close()
        
        # Filter out system tables
        if db_type == "postgresql":
            # Filter out PostgreSQL system tables
            system_tables = ['pg_', 'sql_', 'information_schema', 'system_']
            tables = [table for table in tables if not any(table.startswith(prefix) for prefix in system_tables)]
        elif db_type == "mysql":
            # Filter out MySQL system tables
            system_tables = ['mysql', 'information_schema', 'performance_schema', 'sys']
            tables = [table for table in tables if table not in system_tables]
        
        return {"tables": tables}
    except Exception as e:
        return {"error": str(e)}

@app.post("/db/import_one")
async def db_import_one(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...),
    processing_mode: str = Form("fast"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        # FIXED: Use simple database import (no complex large table handling)
        df = import_table_to_dataframe(conn, table_name)
        conn.close()
        
        file_info = {"source": f"db:{db_type}", "table": table_name, "size": len(df)}
        
        # Route to appropriate pipeline based on processing mode
        if processing_mode == "fast":
            result = run_fast_pipeline(
                df, 
                file_info=file_info,
                use_openai=use_openai,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "fast", "summary": result}
        elif processing_mode == "config1":
            result = run_config1_pipeline(
                df, 
                null_handling="keep",
                fill_value="Unknown",
                chunk_method="recursive",
                chunk_size=400,
                overlap=50,
                model_choice="text-embedding-ada-002" if use_openai else "paraphrase-MiniLM-L6-v2",
                storage_choice="faiss",
                file_info=file_info,
                use_openai=use_openai,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "config1", "summary": result}
        elif processing_mode == "deep":
            result = run_deep_pipeline(
                df,
                null_handling="keep",
                fill_value="Unknown",
                remove_stopwords=False,
                lowercase=True,
                text_processing_option="none",
                chunk_method="recursive",
                chunk_size=400,
                overlap=50,
                model_choice="text-embedding-ada-002" if use_openai else "paraphrase-MiniLM-L6-v2",
                storage_choice="faiss",
                file_info=file_info,
                use_openai=use_openai,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "deep", "summary": result}
        else:
            return {"error": f"Unknown processing mode: {processing_mode}"}
            
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced FAST MODE with Large File & OpenAI Support (FIXED)
# ---------------------------
@app.post("/run_fast")
async def run_fast(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        # Handle database input (FIXED: Simple and working)
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            # FIXED: Use simple database import
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name, "size": len(df)}
            
            result = run_fast_pipeline(
                df, db_type, 
                file_info=file_info,
                use_openai=use_openai,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "fast", "summary": result}
        
        # Handle file input - UPDATED FOR LARGE FILES
        elif file:
            # Create temporary file and stream upload directly to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                # Stream the upload directly to disk (no memory loading)
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                file_size = os.path.getsize(tmp_path)
                
                # Process directly from filesystem
                if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                    result = process_file_direct(
                        tmp_path, 
                        processing_mode="fast",
                        use_openai=use_openai,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                else:
                    # For smaller files, use existing pipeline
                    df = pd.read_csv(tmp_path)
                    file_info = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat()
                    }
                    
                    result = run_fast_pipeline(
                        df, db_type, 
                        file_info=file_info,
                        use_openai=use_openai,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                
                # Add file info to result
                if 'file_info' not in result:
                    result["file_info"] = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat(),
                        "large_file_processed": file_size > LARGE_FILE_THRESHOLD,
                        "turbo_mode": use_turbo,
                        "batch_size": batch_size
                    }
                
                return {"mode": "fast", "summary": result}
                
            finally:
                # Always clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            return {"error": "Either file upload or database parameters required"}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced CONFIG-1 MODE (FIXED)
# ---------------------------
@app.post("/run_config1")
async def run_config1(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        # Handle database input (FIXED: Simple and working)
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            # FIXED: Use simple database import
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name, "size": len(df)}
        
        # Handle file input
        elif file:
            # Create temporary file and stream upload directly to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                # Stream the upload directly to disk (no memory loading)
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                file_size = os.path.getsize(tmp_path)
                
                # Process directly from filesystem for large files
                if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                    result = process_file_direct(
                        tmp_path, 
                        processing_mode="config1",
                        use_openai=use_openai,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                    
                    # Add file info to result
                    if 'file_info' not in result:
                        result["file_info"] = {
                            "filename": file.filename,
                            "file_size": file_size,
                            "upload_time": pd.Timestamp.now().isoformat(),
                            "large_file_processed": True,
                            "turbo_mode": use_turbo,
                            "batch_size": batch_size
                        }
                    
                    return {"mode": "config1", "summary": result}
                else:
                    # For smaller files, use existing pipeline
                    df = pd.read_csv(tmp_path)
                    file_info = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat()
                    }
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            return {"error": "Either file upload or database parameters required"}
        
        # For smaller files or database imports, use the original pipeline
        result = run_config1_pipeline(
            df, null_handling, fill_value, chunk_method,
            chunk_size, overlap, model_choice, storage_choice, 
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        return {"mode": "config1", "summary": result}
    
    except Exception as e:
        return {"error": str(e)}

# Enhanced DEEP CONFIG MODE (FIXED)
# ---------------------------
@app.post("/run_deep")
async def run_deep(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    column_types: str = Form("{}"),
    remove_stopwords: bool = Form(False),
    lowercase: bool = Form(True),
    text_processing_option: str = Form("none"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    document_key_column: str = Form(None),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        # Handle database input (FIXED: Simple and working)
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            # FIXED: Use simple database import
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name, "size": len(df)}
        
        # Handle file input
        elif file:
            # Create temporary file and stream upload directly to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                # Stream the upload directly to disk (no memory loading)
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                file_size = os.path.getsize(tmp_path)
                
                # Process directly from filesystem for large files
                if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                    # For deep mode with large files, we need to handle column types differently
                    # Since we can't easily parse JSON from the filesystem processing
                    result = process_file_direct(
                        tmp_path, 
                        processing_mode="deep",
                        use_openai=use_openai,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                    
                    # Add file info to result
                    if 'file_info' not in result:
                        result["file_info"] = {
                            "filename": file.filename,
                            "file_size": file_size,
                            "upload_time": pd.Timestamp.now().isoformat(),
                            "large_file_processed": True,
                            "turbo_mode": use_turbo,
                            "batch_size": batch_size
                        }
                    
                    return {"mode": "deep", "summary": result}
                else:
                    # For smaller files, use existing pipeline
                    df = pd.read_csv(tmp_path)
                    file_info = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat()
                    }
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            return {"error": "Either file upload or database parameters required"}
        
        # Parse column types from JSON string (only for file uploads)
        try:
            column_types_dict = json.loads(column_types) if file else {}
        except:
            column_types_dict = {}
        
        # For smaller files or database imports, use the original pipeline
        result = run_deep_pipeline(
            df, null_handling, fill_value, remove_stopwords,
            lowercase, text_processing_option, chunk_method,
            chunk_size, overlap, model_choice, storage_choice, 
            column_types=column_types_dict, 
            document_key_column=document_key_column,
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        return {"mode": "deep", "summary": result}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced RETRIEVAL ENDPOINTS
# ---------------------------
@app.post("/retrieve")
async def retrieve(query: str = Form(...), k: int = Form(5)):
    """Retrieve similar chunks after running any pipeline"""
    result = retrieve_similar(query, k)
    return result

@app.post("/v1/retrieve")
async def openai_style_retrieve(
    query: str = Form(...),
    model: str = Form("all-MiniLM-L6-v2"),
    n_results: int = Form(5)
):
    """OpenAI-style retrieval endpoint"""
    result = retrieve_similar(query, n_results)
    
    # Format in OpenAI style
    if "error" in result:
        return {"error": result["error"]}
    
    formatted_results = []
    for res in result["results"]:
        formatted_results.append({
            "object": "retrieval_result",
            "score": res["similarity"],
            "content": res["content"],
            "rank": res["rank"]
        })
    
    return {
        "object": "list",
        "data": formatted_results,
        "model": model,
        "query": query,
        "n_results": n_results
    }

# SYSTEM INFO ENDPOINTS
# ---------------------------
@app.get("/system_info")
async def system_info():
    """Get system information"""
    return get_system_info()

@app.get("/file_info")
async def file_info():
    """Get file information"""
    return get_file_info()

# EXPORT ENDPOINTS
# ---------------------------
@app.get("/export/chunks")
async def export_chunks_file():
    """Export chunks as text file"""
    chunks_text = export_chunks()
    if not chunks_text:
        return {"error": "No chunks available"}
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(chunks_text)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="chunks.txt", media_type="text/plain")

@app.get("/export/embeddings")
async def export_embeddings_file():
    """Export embeddings as numpy file"""
    embeddings = export_embeddings()
    if embeddings is None:
        return {"error": "No embeddings available"}
    
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, embeddings)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="embeddings.npy", media_type="application/octet-stream")

@app.get("/export/embeddings_text")
async def export_embeddings_text_file():
    """Export embeddings as readable text file"""
    embeddings = export_embeddings()
    if embeddings is None:
        return {"error": "No embeddings available"}
    
    # Convert embeddings to readable text format
    embeddings_text = "Embeddings (shape: {}):\n\n".format(embeddings.shape)
    
    for i in range(min(len(embeddings), 1000)):  # Limit to first 1000 for performance
        embeddings_text += "Chunk {}: [".format(i+1)
        # Show first 10 dimensions for readability
        for j in range(min(10, len(embeddings[i]))):
            embeddings_text += "{:.6f}".format(embeddings[i][j])
            if j < min(9, len(embeddings[i])-1):
                embeddings_text += ", "
        embeddings_text += "...]\n"
    
    if len(embeddings) > 1000:
        embeddings_text += "\n... and {} more embeddings".format(len(embeddings) - 1000)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(embeddings_text)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="embeddings.txt", media_type="text/plain")

# NEW: Export Preprocessed Data Endpoint
@app.get("/export/preprocessed")
async def export_preprocessed_file():
    """Export preprocessed data as text file"""
    preprocessed_text = export_preprocessed_data()
    if not preprocessed_text:
        return {"error": "No preprocessed data available"}
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(preprocessed_text)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="preprocessed_data.txt", media_type="text/plain")

# HEALTH CHECK & LARGE FILE SUPPORT INFO
# ---------------------------
@app.get("/")
async def root():
    system_info = get_system_info()
    return {
        "message": "Chunking Optimizer API is running", 
        "version": "2.0",
        "large_file_support": True,
        "max_recommended_file_size": system_info.get("max_recommended_file_size", "N/A"),
        "openai_compatible": True,
        "performance_optimized": True,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "parallel_workers": PARALLEL_WORKERS
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "large_file_support": True, "performance_optimized": True}

@app.get("/capabilities")
async def capabilities():
    """Return API capabilities"""
    return {
        "openai_compatible_endpoints": [
            "/v1/embeddings",
            "/v1/chat/completions", 
            "/v1/retrieve"
        ],
        "large_file_support": True,
        "max_file_size_recommendation": "3GB+",
        "supported_embedding_models": [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2", 
            "text-embedding-ada-002"
        ],
        "batch_processing": True,
        "memory_optimized": True,
        "database_large_table_support": True,
        "performance_features": {
            "turbo_mode": True,
            "parallel_processing": True,
            "optimized_batch_size": 256,
            "caching_system": True
        }
    }

# ---------------------------
# NEW: Large File Upload Endpoint
# ---------------------------
@app.post("/upload_large_file")
async def upload_large_file(
    file: UploadFile = File(...),
    processing_mode: str = Form("fast"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    """Direct large file upload endpoint with disk streaming"""
    try:
        # Create temporary file and stream upload directly to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            # Stream the upload directly to disk (no memory loading)
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            file_size = os.path.getsize(tmp_path)
            
            # Process directly from filesystem
            result = process_file_direct(
                tmp_path, 
                processing_mode=processing_mode,
                use_openai=use_openai,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            
            # Add file info to result
            result["file_info"] = {
                "filename": file.filename,
                "file_size": file_size,
                "upload_time": pd.Timestamp.now().isoformat(),
                "large_file_processed": True,
                "processing_mode": processing_mode,
                "turbo_mode": use_turbo,
                "batch_size": batch_size
            }
            
            return {"mode": processing_mode, "summary": result}
            
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)