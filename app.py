# app.py (Streamlit Frontend) - COMPLETE UPDATED VERSION WITH ENHANCED UI
import streamlit as st
import pandas as pd
import requests
import io
import time
import base64
import os
from datetime import datetime
import json
import tempfile
import shutil

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# ---------- Enhanced Dark Grey Theme with Animations ----------
def load_css():
    """Load custom CSS styles with enhanced animations and dark grey theme"""
    st.markdown(f"""
        <style>
            /* Root variables with dark grey color scheme */
            :root {{
                --primary-gradient: linear-gradient(135deg, #f26f21 0%, #ffa800 100%);
                --success-gradient: linear-gradient(135deg, #48bb78, #38a169);
                --warning-gradient: linear-gradient(135deg, #ed8936, #dd6b20);
                --error-gradient: linear-gradient(135deg, #f56565, #e53e3e);
                --glass-bg: #2d374899;
                --glass-border: rgba(255, 255, 255, 0.15);
                --text-primary: #ffffff;
                --text-secondary: #e2e8f0;
                --text-muted: #a0aec0;
                --border-light: rgba(255, 255, 255, 0.1);
                --bg-light: #4a5568;
                --background-color: #1a202c;
                --card-background: #2d3748;
                --dark-grey-bg: #2d3748;
                --darker-grey-bg: #1a202c;
            }}
            
            /* Main styling with dark grey theme */
            .main {{
                background: var(--darker-grey-bg);
                color: var(--text-primary);
            }}
            
            .stApp {{
                background: linear-gradient(135deg, #1a202c 0%, #2d3748 50%, #4a5568 100%);
                min-height: 100vh;
            }}
            
            /* Enhanced Header animations */
            @keyframes float {{
                0% {{ transform: translateY(0px) rotate(0deg); }}
                50% {{ transform: translateY(-15px) rotate(2deg); }}
                100% {{ transform: translateY(0px) rotate(0deg); }}
            }}
            
            @keyframes glow {{
                0% {{ 
                    box-shadow: 0 0 5px #f26f21, 0 0 10px rgba(242, 111, 33, 0.3);
                    transform: scale(1);
                }}
                50% {{ 
                    box-shadow: 0 0 20px #ffa800, 0 0 30px rgba(255, 168, 0, 0.5);
                    transform: scale(1.02);
                }}
                100% {{ 
                    box-shadow: 0 0 5px #f26f21, 0 0 10px rgba(242, 111, 33, 0.3);
                    transform: scale(1);
                }}
            }}
            
            @keyframes slideIn {{
                0% {{ 
                    transform: translateX(-100%); 
                    opacity: 0; 
                    filter: blur(10px);
                }}
                100% {{ 
                    transform: translateX(0); 
                    opacity: 1;
                    filter: blur(0);
                }}
            }}
            
            @keyframes fadeInUp {{
                0% {{ 
                    opacity: 0; 
                    transform: translateY(30px) scale(0.95);
                    filter: blur(5px);
                }}
                100% {{ 
                    opacity: 1; 
                    transform: translateY(0) scale(1);
                    filter: blur(0);
                }}
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            
            @keyframes shimmer {{
                0% {{ background-position: -200px 0; }}
                100% {{ background-position: 200px 0; }}
            }}
            
            .float-animation {{
                animation: float 4s ease-in-out infinite;
                text-align: center;
            }}
            
            .glow-animation {{
                animation: glow 3s ease-in-out infinite;
            }}
            
            .slide-in {{
                animation: slideIn 0.8s ease-out;
            }}
            
            .fade-in-up {{
                animation: fadeInUp 0.6s ease-out;
            }}
            
            .pulse-animation {{
                animation: pulse 2s ease-in-out infinite;
            }}
            
            .shimmer-effect {{
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                background-size: 200px 100%;
                animation: shimmer 2s infinite;
            }}
            
            /* Enhanced Card styling */
            .feature-card {{
                background: var(--card-background);
                backdrop-filter: blur(15px);
                border: 1px solid var(--glass-border);
                border-radius: 20px;
                padding: 25px;
                margin: 15px 0;
                transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                position: relative;
                overflow: hidden;
            }}
            
            .feature-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                transition: left 0.5s;
            }}
            
            .feature-card:hover::before {{
                left: 100%;
            }}
            
            .feature-card:hover {{
                transform: translateY(-12px) scale(1.03);
                box-shadow: 0 20px 60px rgba(242, 111, 33, 0.4);
                border-color: #f26f21;
            }}
            
            .chart-card {{
                background: var(--card-background);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                padding: 20px;
                margin: 15px 0;
                transition: all 0.4s ease;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }}
            
            .chart-card:hover {{
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 15px 35px rgba(242, 111, 33, 0.3);
                border-color: #ffa800;
            }}
            
            /* Enhanced Button styling */
            .stButton button {{
                background: var(--primary-gradient);
                color: white;
                border: none;
                border-radius: 15px;
                padding: 14px 28px;
                font-weight: 700;
                font-size: 16px;
                transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                box-shadow: 0 6px 20px rgba(242, 111, 33, 0.4);
                position: relative;
                overflow: hidden;
            }}
            
            .stButton button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }}
            
            .stButton button:hover::before {{
                left: 100%;
            }}
            
            .stButton button:hover {{
                transform: translateY(-4px) scale(1.05);
                box-shadow: 0 12px 30px rgba(242, 111, 33, 0.6);
                background: var(--primary-gradient);
            }}
            
            .stButton button:active {{
                transform: translateY(-2px) scale(1.02);
            }}
            
            /* Enhanced Metric card styling */
            .metric-card {{
                background: var(--card-background);
                border-radius: 16px;
                padding: 25px;
                margin: 12px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                border-left: 6px solid #f26f21;
                transition: all 0.4s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--primary-gradient);
                transform: scaleX(0);
                transition: transform 0.3s ease;
            }}
            
            .metric-card:hover::before {{
                transform: scaleX(1);
            }}
            
            .metric-card:hover {{
                transform: translateY(-6px) scale(1.03);
                box-shadow: 0 15px 35px rgba(242, 111, 33, 0.3);
            }}
            
            /* Enhanced Template selector styling */
            .template-option {{
                background: var(--card-background);
                border: 3px solid var(--glass-border);
                border-radius: 16px;
                padding: 20px;
                margin: 10px;
                cursor: pointer;
                transition: all 0.4s ease;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            
            .template-option::after {{
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--primary-gradient);
                transform: scaleX(0);
                transition: transform 0.3s ease;
            }}
            
            .template-option:hover {{
                border-color: #f26f21;
                transform: translateY(-5px) scale(1.05);
                box-shadow: 0 10px 30px rgba(242, 111, 33, 0.3);
            }}
            
            .template-option.selected {{
                border-color: #f26f21;
                background: rgba(242, 111, 33, 0.15);
                box-shadow: 0 8px 25px rgba(242, 111, 33, 0.3);
            }}
            
            .template-option.selected::after {{
                transform: scaleX(1);
            }}
            
            /* Slide preview styling */
            .slide-preview {{
                background: var(--card-background);
                border: 2px solid var(--glass-border);
                border-radius: 20px;
                padding: 25px;
                margin: 15px 0;
                transition: all 0.4s ease;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }}
            
            .slide-preview:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(242, 111, 33, 0.2);
                border-color: #ffa800;
            }}
            
            /* Enhanced navigation styling */
            .nav-item {{
                transition: all 0.3s ease;
                border-radius: 10px;
                margin: 5px 0;
            }}
            
            .nav-item:hover {{
                background: rgba(242, 111, 33, 0.1);
                transform: translateX(5px);
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: var(--darker-grey-bg);
                border-radius: 10px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: var(--primary-gradient);
                border-radius: 10px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(135deg, #ffa800, #f26f21);
            }}
            
            /* Enhanced checkbox styling */
            .stCheckbox label {{
                color: var(--text-primary);
                font-weight: 600;
                font-size: 16px;
                transition: all 0.3s ease;
            }}
            
            .stCheckbox label:hover {{
                color: #ffa800;
                transform: translateX(5px);
            }}
            
            /* Enhanced select box styling */
            .stSelectbox label {{
                color: var(--text-primary);
                font-weight: 600;
            }}
            
            /* Enhanced text input styling */
            .stTextInput input {{
                background: var(--card-background);
                border: 2px solid var(--glass-border);
                border-radius: 12px;
                color: var(--text-primary);
                font-size: 16px;
                padding: 12px;
                transition: all 0.3s ease;
            }}
            
            .stTextInput input:focus {{
                border-color: #f26f21;
                box-shadow: 0 0 0 3px rgba(242, 111, 33, 0.2);
                transform: scale(1.02);
            }}
            
            /* Enhanced file uploader styling */
            .stFileUploader label {{
                color: var(--text-primary);
                font-weight: 600;
                font-size: 16px;
            }}
            
            /* Success message styling */
            .stSuccess {{
                background: var(--success-gradient);
                color: white;
                border-radius: 15px;
                padding: 15px;
                border-left: 6px solid #38a169;
                animation: slideIn 0.5s ease-out;
            }}
            
            /* Info message styling */
            .stInfo {{
                background: var(--card-background);
                color: var(--text-primary);
                border: 2px solid var(--glass-border);
                border-radius: 15px;
                padding: 15px;
                border-left: 6px solid #4299e1;
            }}
            
            /* Expander styling */
            .streamlit-expanderHeader {{
                background: var(--card-background);
                border: 2px solid var(--glass-border);
                border-radius: 15px;
                padding: 18px;
                margin: 8px 0;
                transition: all 0.3s ease;
                font-weight: 600;
            }}
            
            .streamlit-expanderHeader:hover {{
                border-color: #f26f21;
                box-shadow: 0 8px 25px rgba(242, 111, 33, 0.2);
                transform: translateY(-2px);
            }}
            
            /* Progress bar styling */
            .stProgress > div > div {{
                background: var(--primary-gradient);
                border-radius: 10px;
            }}
            
            /* Custom cards for our app */
            .custom-card {{
                background: var(--card-background);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                transition: all 0.4s ease;
                border-left: 6px solid #f26f21;
            }}
            
            .custom-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(242, 111, 33, 0.3);
            }}
            
            .card-title {{
                color: var(--text-primary);
                font-size: 1.3em;
                font-weight: 700;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .card-content {{
                color: var(--text-secondary);
                font-size: 1em;
                line-height: 1.6;
            }}
            
            /* Process steps */
            .process-step {{
                background: var(--card-background);
                padding: 15px;
                border-radius: 12px;
                margin: 8px 0;
                border-left: 6px solid #666;
                transition: all 0.3s ease;
            }}
            
            .process-step.running {{
                border-left-color: #ed8936;
                background: linear-gradient(90deg, var(--card-background), #4a5568);
            }}
            
            .process-step.completed {{
                border-left-color: #48bb78;
                background: linear-gradient(90deg, var(--card-background), #2d4a2d);
            }}
            
            .process-step.pending {{
                border-left-color: #666;
                background: var(--card-background);
            }}
            
            /* Large file warning */
            .large-file-warning {{
                background: var(--warning-gradient);
                color: white;
                padding: 15px;
                border-radius: 12px;
                margin: 10px 0;
                border-left: 6px solid #ed8936;
                animation: pulse 2s infinite;
            }}
            
            /* Scrollable chunk display */
            .scrollable-chunk {{
                background: var(--darker-grey-bg);
                border: 1px solid var(--glass-border);
                border-radius: 8px;
                padding: 12px;
                margin: 5px 0;
                max-height: 300px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 0.85em;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            
            .chunk-header {{
                background: var(--card-background);
                padding: 10px 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                font-weight: bold;
                border-left: 4px solid #f26f21;
            }}
            
            /* Make text areas wider */
            .stTextArea > div > div > textarea {{
                width: 100% !important;
                background: var(--card-background);
                border: 2px solid var(--glass-border);
                border-radius: 12px;
                color: var(--text-primary);
            }}
            
            /* Radio buttons */
            .stRadio > div {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 10px;
            }}
            
            /* Sidebar enhancements */
            .css-1d391kg {{
                background: linear-gradient(180deg, var(--darker-grey-bg) 0%, var(--card-background) 100%) !important;
            }}
            
            /* Dataframe styling */
            .dataframe {{
                background: var(--card-background) !important;
                color: var(--text-primary) !important;
            }}
            
            /* Error message styling */
            .stError {{
                background: var(--error-gradient);
                color: white;
                border-radius: 15px;
                padding: 15px;
                border-left: 6px solid #e53e3e;
            }}
            
            /* Warning message styling */
            .stWarning {{
                background: var(--warning-gradient);
                color: white;
                border-radius: 15px;
                padding: 15px;
                border-left: 6px solid #ed8936;
            }}
        </style>
    """, unsafe_allow_html=True)

# Load CSS when app starts
load_css()

# ---------- API Client Functions ----------
def call_fast_api(file_path: str, filename: str, db_type: str, db_config: dict = None, 
                  use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                  process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Send file directly from filesystem path"""
    try:
        with open(file_path, 'rb') as f:
            if db_config and db_config.get('use_db'):
                data = {
                    "db_type": db_config["db_type"],
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "username": db_config["username"],
                    "password": db_config["password"],
                    "database": db_config["database"],
                    "table_name": db_config["table_name"],
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                }
                response = requests.post(f"{API_BASE_URL}/run_fast", data=data)
            else:
                files = {"file": (filename, f, "text/csv")}
                data = {
                    "db_type": db_type,
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                }
                response = requests.post(f"{API_BASE_URL}/run_fast", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_config1_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                    use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                    process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Send file directly from filesystem path"""
    try:
        with open(file_path, 'rb') as f:
            if db_config and db_config.get('use_db'):
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "db_type": db_config["db_type"],
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "username": db_config["username"],
                    "password": db_config["password"],
                    "database": db_config["database"],
                    "table_name": db_config["table_name"],
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                })
                response = requests.post(f"{API_BASE_URL}/run_config1", data=data)
            else:
                files = {"file": (filename, f, "text/csv")}
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                })
                response = requests.post(f"{API_BASE_URL}/run_config1", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_deep_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                 use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                 process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Send file directly from filesystem path"""
    try:
        with open(file_path, 'rb') as f:
            if db_config and db_config.get('use_db'):
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "db_type": db_config["db_type"],
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "username": db_config["username"],
                    "password": db_config["password"],
                    "database": db_config["database"],
                    "table_name": db_config["table_name"],
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                })
                response = requests.post(f"{API_BASE_URL}/run_deep", data=data)
            else:
                files = {"file": (filename, f, "text/csv")}
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                })
                response = requests.post(f"{API_BASE_URL}/run_deep", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_retrieve_api(query: str, k: int = 5):
    data = {"query": query, "k": k}
    response = requests.post(f"{API_BASE_URL}/retrieve", data=data)
    return response.json()

def call_openai_retrieve_api(query: str, model: str = "all-MiniLM-L6-v2", n_results: int = 5):
    data = {"query": query, "model": model, "n_results": n_results}
    response = requests.post(f"{API_BASE_URL}/v1/retrieve", data=data)
    return response.json()

def call_openai_embeddings_api(text: str, model: str = "text-embedding-ada-002", 
                              openai_api_key: str = None, openai_base_url: str = None):
    data = {
        "model": model,
        "input": text,
        "openai_api_key": openai_api_key,
        "openai_base_url": openai_base_url
    }
    response = requests.post(f"{API_BASE_URL}/v1/embeddings", data=data)
    return response.json()

def get_system_info_api():
    response = requests.get(f"{API_BASE_URL}/system_info")
    return response.json()

def get_file_info_api():
    response = requests.get(f"{API_BASE_URL}/file_info")
    return response.json()

def get_capabilities_api():
    response = requests.get(f"{API_BASE_URL}/capabilities")
    return response.json()

def download_file(url: str, filename: str):
    response = requests.get(f"{API_BASE_URL}{url}")
    return response.content

def download_embeddings_text():
    """Download embeddings in text format"""
    response = requests.get(f"{API_BASE_URL}/export/embeddings_text")
    return response.content

# Database helper functions
def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

# ---------- Large File Helper Functions ----------
def is_large_file(file_size: int, threshold_mb: int = 100) -> bool:
    """Check if file is considered large"""
    return file_size > threshold_mb * 1024 * 1024

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def handle_file_upload(uploaded_file):
    """
    Safely handle file uploads by streaming to disk (no memory loading)
    Returns temporary file path and file info
    """
    # Create temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        # Stream the uploaded file directly to disk
        shutil.copyfileobj(uploaded_file, tmp_file)
        temp_path = tmp_file.name
    
    # Get file size from disk
    file_size = os.path.getsize(temp_path)
    file_size_str = format_file_size(file_size)
    
    file_info = {
        "name": uploaded_file.name,
        "size": file_size_str,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Temporary storage",
        "temp_path": temp_path
    }
    
    return temp_path, file_info

# ---------- Scrollable Chunk Display Function ----------
def display_scrollable_chunk(result, chunk_index):
    """Display chunk content in a scrollable container"""
    similarity_color = "#48bb78" if result['similarity'] > 0.7 else "#ed8936" if result['similarity'] > 0.4 else "#f56565"
    
    # Create a unique key for the expander
    expander_key = f"chunk_{chunk_index}_{result['rank']}"
    
    with st.expander(f"📄 Rank #{result['rank']} (Similarity: {result['similarity']:.3f})", expanded=False):
        # Header with similarity score
        st.markdown(f"""
        <div style="background: #2d3748; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 6px solid {similarity_color};">
            <strong>Rank:</strong> {result['rank']} | 
            <strong>Similarity:</strong> {result['similarity']:.3f} | 
            <strong>Distance:</strong> {result.get('distance', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        # Scrollable content area
        st.markdown("""
        <div class="chunk-header">
            📋 Chunk Content (Scrollable)
        </div>
        """, unsafe_allow_html=True)
        
        # Use text_area for scrollable content but make it read-only
        content = result['content']
        
        # Create a scrollable text area
        st.text_area(
            "Chunk Content",
            value=content,
            height=300,
            key=f"chunk_content_{chunk_index}",
            disabled=True,
            label_visibility="collapsed"
        )

# ---------- Streamlit App ----------
st.set_page_config(page_title="Chunking Optimizer", layout="wide", page_icon="📦")

# Enhanced header with animations
st.markdown("""
<div class="float-animation" style="background: var(--primary-gradient); padding: 40px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 15px 35px rgba(242, 111, 33, 0.4);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🚀 Chunking Optimizer v2.0</h1>
    <p style="color: white; text-align: center; margin: 15px 0 0 0; font-size: 1.4em; font-weight: 500;">Advanced Text Processing + 3GB File Support + Performance Optimized</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "api_results" not in st.session_state:
    st.session_state.api_results = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = None
if "process_status" not in st.session_state:
    st.session_state.process_status = {
        "preprocessing": "pending",
        "chunking": "pending", 
        "embedding": "pending",
        "storage": "pending",
        "retrieval": "pending"
    }
if "process_timings" not in st.session_state:
    st.session_state.process_timings = {}
if "file_info" not in st.session_state:
    st.session_state.file_info = {}
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "column_types" not in st.session_state:
    st.session_state.column_types = {}
if "preview_df" not in st.session_state:
    st.session_state.preview_df = None
if "text_processing_option" not in st.session_state:
    st.session_state.text_processing_option = "none"
if "preview_updated" not in st.session_state:
    st.session_state.preview_updated = False
if "use_openai" not in st.session_state:
    st.session_state.use_openai = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "openai_base_url" not in st.session_state:
    st.session_state.openai_base_url = ""
if "process_large_files" not in st.session_state:
    st.session_state.process_large_files = True
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "use_turbo" not in st.session_state:
    st.session_state.use_turbo = True
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 256

# Sidebar with process tracking and system info
with st.sidebar:
    st.markdown("""
    <div class="glow-animation" style="background: var(--primary-gradient); padding: 25px; border-radius: 15px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0; font-size: 1.5em;">⚡ Process Tracker</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # API connection test
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.success("✅ API Connected")
        
        # Show capabilities
        capabilities = get_capabilities_api()
        if capabilities.get('large_file_support'):
            st.info("🚀 3GB+ File Support")
        if capabilities.get('performance_features', {}).get('turbo_mode'):
            st.info("⚡ Turbo Mode Available")
            
    except:
        st.error("❌ API Not Connected")
    
    st.markdown("---")
    
    # OpenAI Configuration
    with st.expander("🤖 OpenAI Configuration", expanded=False):
        st.session_state.use_openai = st.checkbox("Use OpenAI API", value=st.session_state.use_openai)
        
        if st.session_state.use_openai:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", 
                                                          value=st.session_state.openai_api_key,
                                                          type="password",
                                                          help="Your OpenAI API key")
            st.session_state.openai_base_url = st.text_input("OpenAI Base URL (optional)", 
                                                           value=st.session_state.openai_base_url,
                                                           placeholder="https://api.openai.com/v1",
                                                           help="Custom OpenAI-compatible API endpoint")
            
            if st.session_state.openai_api_key:
                st.success("✅ OpenAI API Configured")
            else:
                st.warning("⚠️ Please enter OpenAI API Key")
    
    # Large File Configuration
    with st.expander("💾 Large File Settings", expanded=False):
        st.session_state.process_large_files = st.checkbox(
            "Enable Large File Processing", 
            value=st.session_state.process_large_files,
            help="Process files larger than 100MB in batches to avoid memory issues"
        )
        
        if st.session_state.process_large_files:
            st.info("""**Large File Features:**
            - Direct disk streaming (no memory overload)
            - Batch processing for memory efficiency
            - Automatic chunking for files >100MB
            - Progress tracking for large datasets
            - Support for 3GB+ files
            """)
    
    # Process steps display
    st.markdown("### ⚙️ Processing Steps")
    
    steps = [
        ("preprocessing", "🧹 Preprocessing"),
        ("chunking", "📦 Chunking"), 
        ("embedding", "🤖 Embedding"),
        ("storage", "💾 Vector DB"),
        ("retrieval", "🔍 Retrieval")
    ]
    
    for step_key, step_name in steps:
        status = st.session_state.process_status.get(step_key, "pending")
        timing = st.session_state.process_timings.get(step_key, "")
        
        if status == "completed":
            icon = "✅"
            color = "completed"
            timing_display = f"({timing})" if timing else ""
        elif status == "running":
            icon = "🟠"
            color = "running"
            timing_display = ""
        else:
            icon = "⚪"
            color = "pending"
            timing_display = ""
        
        st.markdown(f"""
        <div class="process-step {color}">
            {icon} <strong>{step_name}</strong> {timing_display}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Information
    st.markdown("### 💻 System Information")
    try:
        system_info = get_system_info_api()
        st.write(f"**Memory Usage:** {system_info.get('memory_usage', 'N/A')}")
        st.write(f"**Available Memory:** {system_info.get('available_memory', 'N/A')}")
        st.write(f"**Total Memory:** {system_info.get('total_memory', 'N/A')}")
        st.write(f"**Batch Size:** {system_info.get('embedding_batch_size', 'N/A')}")
        if system_info.get('large_file_support'):
            st.write(f"**Max File Size:** {system_info.get('max_recommended_file_size', 'N/A')}")
    except:
        st.write("**Memory Usage:** N/A")
        st.write("**Available Memory:** N/A")
        st.write("**Total Memory:** N/A")
    
    # File Information
    st.markdown("### 📁 File Information")
    if st.session_state.file_info:
        file_info = st.session_state.file_info
        st.write(f"**File Name:** {file_info.get('name', 'N/A')}")
        st.write(f"**File Size:** {file_info.get('size', 'N/A')}")
        st.write(f"**Upload Time:** {file_info.get('upload_time', 'N/A')}")
        if file_info.get('large_file_processed'):
            st.success("✅ Large File Optimized")
        if file_info.get('turbo_mode'):
            st.success("⚡ Turbo Mode Enabled")
    else:
        try:
            file_info = get_file_info_api()
            if file_info and 'filename' in file_info:
                st.write(f"**File Name:** {file_info.get('filename', 'N/A')}")
                st.write(f"**File Size:** {file_info.get('file_size', 0) / 1024:.2f} KB")
                st.write(f"**Upload Time:** {file_info.get('upload_time', 'N/A')}")
                st.write(f"**File Location:** Backend storage")
        except:
            st.write("**File Info:** Not available")
    
    st.markdown("---")
    
    if st.session_state.api_results:
        st.markdown("### 📊 Last Results")
        result = st.session_state.api_results
        st.write(f"**Mode:** {result.get('mode', 'N/A')}")
        if 'summary' in result:
            st.write(f"**Chunks:** {result['summary'].get('chunks', 'N/A')}")
            st.write(f"**Storage:** {result['summary'].get('stored', 'N/A')}")
            st.write(f"**Model:** {result['summary'].get('embedding_model', 'N/A')}")
            if result['summary'].get('turbo_mode'):
                st.success("⚡ Turbo Mode Used")
            if 'conversion_results' in result['summary']:
                conv_results = result['summary']['conversion_results']
                if conv_results:
                    st.write(f"**Type Conversions:** {len(conv_results.get('successful', []))} successful")
            if result['summary'].get('retrieval_ready'):
                st.success("🔍 Retrieval Ready")
            if result['summary'].get('large_file_processed'):
                st.success("🚀 Large File Optimized")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        # Clean up temporary files
        if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
            os.unlink(st.session_state.temp_file_path)
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Mode selection with enhanced cards
st.markdown("## 🎯 Choose Processing Mode")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="template-option {'selected' if st.session_state.current_mode == 'fast' else ''}" onclick="this.classList.toggle('selected')">
        <h3>⚡ Fast Mode</h3>
        <p>Quick processing with optimized defaults</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Fast Mode", key="fast_mode_btn", use_container_width=True):
        st.session_state.current_mode = "fast"
        st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
        st.rerun()

with col2:
    st.markdown("""
    <div class="template-option {'selected' if st.session_state.current_mode == 'config1' else ''}" onclick="this.classList.toggle('selected')">
        <h3>⚙️ Config-1 Mode</h3>
        <p>Balanced customization and performance</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Config-1 Mode", key="config1_mode_btn", use_container_width=True):
        st.session_state.current_mode = "config1"
        st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
        st.rerun()

with col3:
    st.markdown("""
    <div class="template-option {'selected' if st.session_state.current_mode == 'deep' else ''}" onclick="this.classList.toggle('selected')">
        <h3>🔬 Deep Config Mode</h3>
        <p>Advanced customization with full control</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Deep Config Mode", key="deep_mode_btn", use_container_width=True):
        st.session_state.current_mode = "deep"
        st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
        st.rerun()

if st.session_state.current_mode:
    st.success(f"**Selected: {st.session_state.current_mode.upper()} MODE** • {'⚡ Turbo Enabled' if st.session_state.use_turbo else 'Normal Mode'} • Batch Size: {st.session_state.batch_size}")

# Mode-specific processing
if st.session_state.current_mode:
    if st.session_state.current_mode == "fast":
        st.markdown("### ⚡ Fast Mode Configuration")
        
        # Input source selection
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="fast_input_source")
        
        if input_source == "📁 Upload CSV File":
            st.markdown("#### 📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="fast_file_upload")
            
            if uploaded_file is not None:
                # Use filesystem upload method
                with st.spinner("🔄 Streaming file to disk..."):
                    temp_path, file_info = handle_file_upload(uploaded_file)
                    st.session_state.temp_file_path = temp_path
                    st.session_state.file_info = file_info
                
                file_size_str = file_info["size"]
                file_size_bytes = os.path.getsize(temp_path)
                
                # Check if file is large
                if is_large_file(file_size_bytes):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ **{uploaded_file.name}** loaded! ({file_size_str})")
                
            use_db_config = None
            
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="fast_db_type")
                host = st.text_input("Host", "localhost", key="fast_host")
                port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="fast_port")
            
            with col2:
                username = st.text_input("Username", key="fast_username")
                password = st.text_input("Password", type="password", key="fast_password")
                database = st.text_input("Database", key="fast_database")
            
            # Test connection and get tables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Test Connection", key="fast_test_conn", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        res = db_test_connection_api({
                            "db_type": db_type,
                            "host": host,
                            "port": port,
                            "username": username,
                            "password": password,
                            "database": database,
                        })
                        if res.get("status") == "success":
                            st.success("✅ Connection successful!")
                        else:
                            st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("📋 List Tables", key="fast_list_tables", use_container_width=True):
                    with st.spinner("Fetching tables..."):
                        res = db_list_tables_api({
                            "db_type": db_type,
                            "host": host,
                            "port": port,
                            "username": username,
                            "password": password,
                            "database": database,
                        })
                        st.session_state["fast_db_tables"] = res.get("tables", [])
                        if st.session_state["fast_db_tables"]:
                            st.success(f"✅ Found {len(st.session_state['fast_db_tables'])} tables")
                        else:
                            st.warning("⚠️ No tables found")
            
            tables = st.session_state.get("fast_db_tables", [])
            if tables:
                table_name = st.selectbox("Select Table", tables, key="fast_table_select")
                use_db_config = {
                    "use_db": True,
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name
                }
            else:
                use_db_config = None
                st.info("👆 Test connection and list tables first")
        
        # FAST MODE DEFAULTS - No user configuration needed
        # Auto-enable turbo mode and set batch size to 256
        st.session_state.use_turbo = True
        st.session_state.batch_size = 256
        
        # Display Fast Mode pipeline with enhanced card
        processing_type = "Parallel processing" if st.session_state.use_turbo else "Sequential processing"
        
        st.markdown(f"""
        <div class="feature-card">
            <div class="card-title">🚀 Fast Mode Pipeline</div>
            <div class="card-content">
                • Optimized preprocessing for speed<br>
                • Semantic clustering chunking<br>
                • paraphrase-MiniLM-L6-v2 embedding model<br>
                • Batch embedding with size {st.session_state.batch_size}<br>
                • {processing_type}<br>
                • FAISS storage for fast retrieval<br>
                • 3GB+ file support with disk streaming<br>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Fast Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Fast Mode pipeline..."):
                try:
                    # Update process status
                    st.session_state.process_status["preprocessing"] = "running"
                    
                    if input_source == "📁 Upload CSV File":
                        result = call_fast_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            "sqlite",
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    else:
                        result = call_fast_api(
                            None, None, "sqlite", use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    
                    # Update process status
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    
                    # Show performance results
                    if 'summary' in result:
                        if result['summary'].get('large_file_processed'):
                            st.success("✅ Large file processed efficiently with disk streaming!")
                        elif result['summary'].get('turbo_mode'):
                            st.success("⚡ Turbo mode completed successfully!")
                        else:
                            st.success("✅ Fast pipeline completed successfully!")
                    
                    # Show retrieval section immediately
                    st.session_state.process_status["retrieval"] = "completed"
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                        os.unlink(st.session_state.temp_file_path)
                        st.session_state.temp_file_path = None

    elif st.session_state.current_mode == "config1":
        st.markdown("### ⚙️ Config-1 Mode Configuration")
        
        # Input source selection
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="config1_input_source")
        
        if input_source == "📁 Upload CSV File":
            st.markdown("#### 📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="config1_file_upload")
            
            if uploaded_file is not None:
                # Use filesystem upload method
                with st.spinner("🔄 Streaming file to disk..."):
                    temp_path, file_info = handle_file_upload(uploaded_file)
                    st.session_state.temp_file_path = temp_path
                    st.session_state.file_info = file_info
                
                file_size_str = file_info["size"]
                file_size_bytes = os.path.getsize(temp_path)
                
                # Check if file is large
                if is_large_file(file_size_bytes):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ **{uploaded_file.name}** loaded! ({file_size_str})")
                
            use_db_config = None
            
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="config1_db_type")
                host = st.text_input("Host", "localhost", key="config1_host")
                port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="config1_port")
            
            with col2:
                username = st.text_input("Username", key="config1_username")
                password = st.text_input("Password", type="password", key="config1_password")
                database = st.text_input("Database", key="config1_database")
            
            # Test connection and get tables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Test Connection", key="config1_test_conn", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        res = db_test_connection_api({
                            "db_type": db_type,
                            "host": host,
                            "port": port,
                            "username": username,
                            "password": password,
                            "database": database,
                        })
                        if res.get("status") == "success":
                            st.success("✅ Connection successful!")
                        else:
                            st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("📋 List Tables", key="config1_list_tables", use_container_width=True):
                    with st.spinner("Fetching tables..."):
                        res = db_list_tables_api({
                            "db_type": db_type,
                            "host": host,
                            "port": port,
                            "username": username,
                            "password": password,
                            "database": database,
                        })
                        st.session_state["config1_db_tables"] = res.get("tables", [])
                        if st.session_state["config1_db_tables"]:
                            st.success(f"✅ Found {len(st.session_state['config1_db_tables'])} tables")
                        else:
                            st.warning("⚠️ No tables found")
            
            tables = st.session_state.get("config1_db_tables", [])
            if tables:
                table_name = st.selectbox("Select Table", tables, key="config1_table_select")
                use_db_config = {
                    "use_db": True,
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name
                }
            else:
                use_db_config = None
                st.info("👆 Test connection and list tables first")
        
        # Config-1 parameters
        st.markdown("#### ⚙️ Configuration Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🧹 Preprocessing")
            null_handling = st.selectbox("Null value handling", ["keep", "drop", "fill", "mean", "median", "mode"], key="config1_null")
            fill_value = st.text_input("Fill value", "Unknown", key="config1_fill") if null_handling == "fill" else None
            
            st.markdown("##### 📦 Chunking")
            chunk_method = st.selectbox("Chunking method", ["fixed", "recursive", "semantic"], key="config1_chunk")
            
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.slider("Chunk size", 100, 2000, 800, key="config1_size")
                overlap = st.slider("Overlap", 0, 500, 20, key="config1_overlap")
        
        with col2:
            st.markdown("##### 🤖 Embedding")
            model_choice = st.selectbox("Embedding model", 
                                      ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "text-embedding-ada-002"],
                                      key="config1_model")
            
            st.markdown("##### 💾 Storage")
            storage_choice = st.selectbox("Vector storage", ["faiss", "chromadb"], key="config1_storage")
        
        # Performance Configuration for Config1 Mode
        st.markdown("#### ⚡ Performance Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.use_turbo = st.checkbox(
                "Enable Turbo Mode", 
                value=st.session_state.use_turbo,
                help="Faster processing with parallel operations"
            )
        
        with col2:
            st.session_state.batch_size = st.slider(
                "Embedding Batch Size",
                min_value=64,
                max_value=512,
                value=st.session_state.batch_size,
                step=64,
                help="Larger batches = faster processing (requires more memory)"
            )
        
        if st.session_state.use_turbo:
            st.success("✅ Turbo Mode: 2-3x Faster Processing")
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Config-1 Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Config-1 pipeline..."):
                try:
                    # Update process status
                    st.session_state.process_status["preprocessing"] = "running"
                    
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value if fill_value else "Unknown",
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 800,
                        "overlap": overlap if 'overlap' in locals() else 20,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    if input_source == "📁 Upload CSV File":
                        result = call_config1_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            config,
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    else:
                        result = call_config1_api(
                            None, None, config, use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    
                    # Show performance results
                    if 'summary' in result:
                        if result['summary'].get('large_file_processed'):
                            st.success("✅ Large file processed efficiently with disk streaming!")
                        elif result['summary'].get('turbo_mode'):
                            st.success("⚡ Turbo mode completed successfully!")
                        else:
                            st.success("✅ Config-1 pipeline completed successfully!")
                    
                    # Show retrieval section immediately
                    st.session_state.process_status["retrieval"] = "completed"
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                        os.unlink(st.session_state.temp_file_path)
                        st.session_state.temp_file_path = None

    elif st.session_state.current_mode == "deep":
        st.markdown("### 🔬 Deep Config Mode Configuration")
        
        # Input source selection
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="deep_input_source")
        
        if input_source == "📁 Upload CSV File":
            st.markdown("#### 📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="deep_file_upload")
            
            if uploaded_file is not None:
                # Use filesystem upload method
                with st.spinner("🔄 Streaming file to disk..."):
                    temp_path, file_info = handle_file_upload(uploaded_file)
                    st.session_state.temp_file_path = temp_path
                    st.session_state.file_info = file_info
                
                file_size_str = file_info["size"]
                file_size_bytes = os.path.getsize(temp_path)
                
                # Check if file is large
                if is_large_file(file_size_bytes):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Read the CSV file for preview and column type analysis
                try:
                    df = pd.read_csv(temp_path)
                    st.session_state.current_df = df
                    # Initialize preview only once
                    if "preview_df" not in st.session_state or st.session_state.preview_df is None:
                        st.session_state.preview_df = df.head(5).copy()
                    st.success(f"✅ **{uploaded_file.name}** loaded! ({len(df)} rows, {len(df.columns)} columns, {file_size_str})")
                    
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                
            use_db_config = None
            
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="deep_db_type")
                host = st.text_input("Host", "localhost", key="deep_host")
                port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="deep_port")
            
            with col2:
                username = st.text_input("Username", key="deep_username")
                password = st.text_input("Password", type="password", key="deep_password")
                database = st.text_input("Database", key="deep_database")
            
            # Test connection and get tables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Test Connection", key="deep_test_conn", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        res = db_test_connection_api({
                            "db_type": db_type,
                            "host": host,
                            "port": port,
                            "username": username,
                            "password": password,
                            "database": database,
                        })
                        if res.get("status") == "success":
                            st.success("✅ Connection successful!")
                        else:
                            st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("📋 List Tables", key="deep_list_tables", use_container_width=True):
                    with st.spinner("Fetching tables..."):
                        res = db_list_tables_api({
                            "db_type": db_type,
                            "host": host,
                            "port": port,
                            "username": username,
                            "password": password,
                            "database": database,
                        })
                        st.session_state["deep_db_tables"] = res.get("tables", [])
                        if st.session_state["deep_db_tables"]:
                            st.success(f"✅ Found {len(st.session_state['deep_db_tables'])} tables")
                        else:
                            st.warning("⚠️ No tables found")
            
            tables = st.session_state.get("deep_db_tables", [])
            if tables:
                table_name = st.selectbox("Select Table", tables, key="deep_table_select")
                use_db_config = {
                    "use_db": True,
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name
                }
                
            else:
                use_db_config = None
                st.info("👆 Test connection and list tables first")
        
        # Deep config parameters
        st.markdown("#### 🔧 Configuration Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🧹 Preprocessing")
            null_handling = st.selectbox("Null value handling", ["keep", "drop", "fill", "mean", "median", "mode"], key="deep_null")
            fill_value = st.text_input("Fill value", "Unknown", key="deep_fill") if null_handling == "fill" else None
            
            # Enhanced Column Data Type Conversion (only for file uploads)
            if input_source == "📁 Upload CSV File" and st.session_state.current_df is not None:
                st.markdown("##### 🔄 Column Data Type Conversion")
                st.info("Convert column types before processing:")
                
                df = st.session_state.current_df
                preview_df = st.session_state.preview_df.copy() if st.session_state.preview_df is not None else df.head(5).copy()
                column_types = st.session_state.column_types.copy()
                
                # Create a clean preview table with data types
                st.markdown("**Preview (First 5 rows):**")
                
                # Display column headers with data type selection
                for col in preview_df.columns:
                    current_type = str(preview_df[col].dtype)
                    default_idx = 0
                    type_options = ["keep", "string", "numeric", "integer", "float", "datetime", "boolean", "category"]
                    
                    # Set default based on current conversion
                    if col in column_types:
                        default_idx = type_options.index(column_types[col])
                    
                    new_type = st.selectbox(
                        f"**{col}** › Current: `{current_type}`",
                        type_options,
                        index=default_idx,
                        key=f"col_type_{col}"
                    )
                    
                    if new_type != "keep":
                        column_types[col] = new_type
                        # Apply conversion to preview
                        try:
                            if new_type == 'string':
                                preview_df[col] = preview_df[col].astype(str)
                            elif new_type == 'numeric':
                                preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce')
                            elif new_type == 'integer':
                                preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce').fillna(0).astype(int)
                            elif new_type == 'float':
                                preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce').astype(float)
                            elif new_type == 'datetime':
                                preview_df[col] = pd.to_datetime(preview_df[col], errors='coerce')
                            elif new_type == 'boolean':
                                if preview_df[col].dtype == 'object':
                                    true_values = ['true', 'yes', '1', 't', 'y']
                                    preview_df[col] = preview_df[col].astype(str).str.lower().isin(true_values)
                            elif new_type == 'category':
                                preview_df[col] = preview_df[col].astype('category')
                        except Exception as e:
                            st.error(f"Error converting {col}: {str(e)}")
                    elif col in column_types:
                        # Remove from conversions if set to "keep"
                        del column_types[col]
                
                # Display the updated preview
                st.dataframe(preview_df, use_container_width=True)
                
                st.session_state.column_types = column_types
                st.session_state.preview_df = preview_df
                
                if column_types:
                    st.success(f"🎯 {len(column_types)} columns will be converted")
                else:
                    st.info("No column type conversions selected")
            
            st.markdown("##### 🧠 Text Processing")
            remove_stopwords = st.checkbox("Remove stopwords", key="deep_stop")
            lowercase = st.checkbox("Convert to lowercase + clean text", value=True, key="deep_lower")
            
            # Radio button for stemming vs lemmatization (mutually exclusive)
            text_processing_option = st.radio(
                "Advanced text processing:",
                ["none", "stemming", "lemmatization"],
                index=0,
                key="deep_text_processing"
            )
            
        with col2:
            st.markdown("##### 📦 Chunking")
            chunk_method = st.selectbox("Chunking method", ["fixed", "recursive", "semantic", "document"], key="deep_chunk")
            
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.slider("Chunk size", 100, 2000, 800, key="deep_size")
                overlap = st.slider("Overlap", 0, 500, 20, key="deep_overlap")
            
            # Document chunking column selection - show for both file and database
            if chunk_method == "document":
                if st.session_state.current_df is not None:
                    available_columns = st.session_state.current_df.columns.tolist()
                    document_key_column = st.selectbox(
                        "Select column for grouping:",
                        available_columns,
                        key="deep_document_column"
                    )
                    st.info(f"Chunks will be grouped by: **{document_key_column}**")
                else:
                    document_key_column = st.text_input(
                        "Enter column name for grouping:",
                        key="deep_document_column_text"
                    )
                    if document_key_column:
                        st.info(f"Chunks will be grouped by: **{document_key_column}**")
            
            st.markdown("##### 🤖 Embedding & Storage")
            model_choice = st.selectbox("Embedding model", 
                                      ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "text-embedding-ada-002"],
                                      key="deep_model")
            storage_choice = st.selectbox("Vector storage", ["faiss", "chromadb"], key="deep_storage")
        
        # Performance Configuration for Deep Mode
        st.markdown("#### ⚡ Performance Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.use_turbo = st.checkbox(
                "Enable Turbo Mode", 
                value=st.session_state.use_turbo,
                help="Faster processing with parallel operations"
            )
        
        with col2:
            st.session_state.batch_size = st.slider(
                "Embedding Batch Size",
                min_value=64,
                max_value=512,
                value=st.session_state.batch_size,
                step=64,
                help="Larger batches = faster processing (requires more memory)"
            )
        
        if st.session_state.use_turbo:
            st.success("✅ Turbo Mode: 2-3x Faster Processing")
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Deep Config Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Deep Config pipeline..."):
                try:
                    # Update process status
                    st.session_state.process_status["preprocessing"] = "running"
                    
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value if fill_value else "Unknown",
                        "remove_stopwords": remove_stopwords,
                        "lowercase": lowercase,
                        "text_processing_option": text_processing_option,
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 800,
                        "overlap": overlap if 'overlap' in locals() else 20,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    # Add column types only for file uploads (not for database)
                    if input_source == "📁 Upload CSV File":
                        config["column_types"] = json.dumps(st.session_state.column_types)
                    
                    # Add document key column for document chunking
                    if chunk_method == "document":
                        if 'document_key_column' in locals() and document_key_column:
                            config["document_key_column"] = document_key_column
                        elif st.session_state.current_df is not None and len(st.session_state.current_df.columns) > 0:
                            # Use first column as default
                            config["document_key_column"] = st.session_state.current_df.columns[0]
                    
                    if input_source == "📁 Upload CSV File":
                        result = call_deep_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            config,
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    else:
                        result = call_deep_api(
                            None, None, config, use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    
                    # Show conversion results if available
                    if 'summary' in result and 'conversion_results' in result['summary']:
                        conv_results = result['summary']['conversion_results']
                        if conv_results:
                            st.success(f"✅ Column type conversion: {len(conv_results.get('successful', []))} successful")
                            if conv_results.get('failed'):
                                st.warning(f"⚠️ {len(conv_results['failed'])} conversions failed")
                    
                    # Show performance results
                    if 'summary' in result:
                        if result['summary'].get('large_file_processed'):
                            st.success("✅ Large file processed efficiently with disk streaming!")
                        elif result['summary'].get('turbo_mode'):
                            st.success("⚡ Turbo mode completed successfully!")
                        else:
                            st.success("✅ Deep Config pipeline completed successfully!")
                    
                    # Show retrieval section immediately
                    st.session_state.process_status["retrieval"] = "completed"
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                        os.unlink(st.session_state.temp_file_path)
                        st.session_state.temp_file_path = None

# Vector Retrieval Section with Scrollable Chunks
if st.session_state.api_results and st.session_state.api_results.get('summary', {}).get('retrieval_ready'):
    st.markdown("---")
    st.markdown("## 🔍 Semantic Search (Vector DB)")
    st.markdown("Search for similar content using semantic similarity")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        vector_query = st.text_input("Enter semantic search query:", placeholder="Search for similar content...", key="vector_query")
    with col2:
        k = st.slider("Top K results", 1, 10, 3, key="vector_k")
    
    if vector_query:
        with st.spinner("Searching..."):
            try:
                st.session_state.process_status["retrieval"] = "running"
                retrieval_result = call_retrieve_api(vector_query, k)
                st.session_state.process_status["retrieval"] = "completed"
                st.session_state.retrieval_results = retrieval_result
                
                if "error" in retrieval_result:
                    st.error(f"Retrieval error: {retrieval_result['error']}")
                else:
                    st.success(f"✅ Found {len(retrieval_result['results'])} results")
                    
                    # Display each result with scrollable chunk content
                    for i, result in enumerate(retrieval_result['results']):
                        display_scrollable_chunk(result, i)
                        
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")

# Export Section
if st.session_state.api_results:
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Download Chunks")
        if st.button("📄 Export Chunks as TXT", use_container_width=True):
            try:
                chunks_content = download_file("/export/chunks", "chunks.txt")
                st.download_button(
                    label="⬇️ Download Chunks",
                    data=chunks_content,
                    file_name="chunks.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting chunks: {str(e)}")
    
    with col2:
        st.markdown("#### 📥 Download Embeddings")
        if st.button("🔢 Export Embeddings as TXT", use_container_width=True):
            try:
                embeddings_content = download_embeddings_text()
                st.download_button(
                    label="⬇️ Download Embeddings",
                    data=embeddings_content,
                    file_name="embeddings.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting embeddings: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; font-size: 0.9em; padding: 20px;">
    <p>📦 Chunking Optimizer v2.0 • FastAPI + Streamlit • 3GB+ File Support • Performance Optimized</p>
    <p><strong>🚀 Enhanced with Turbo Mode & Parallel Processing • 📜 Scrollable Chunk Display</strong></p>
</div>
""", unsafe_allow_html=True)