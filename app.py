import asyncio
import base64
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()

def load_icon(name):
    with open(f"assets/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode()

ICON_PDF = load_icon("pdf.png")
ICON_UPLOAD = load_icon("upload.png")
ICON_CHAT = load_icon("chat.png")
ICON_CONF = load_icon("confidence.png")
ICON_OK = load_icon("tick.png")
ICON_WARN = load_icon("warning.png")
ICON_SRC = load_icon("sources.png")
ICON_TOP = load_icon("top.png")


st.set_page_config(
    page_title="ContextClerk",
    page_icon="assets/app.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)



st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Google+Sans:wght@400;500;700&display=swap');

        /* Global Reset & Dark Mode */
        .stApp {
            background-color: #131314; /* Deep dark gray/black */
            color: #E3E3E3;
            font-family: 'Google Sans', 'Roboto', sans-serif;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
            font-family: 'Google Sans', sans-serif;
            font-weight: 500;
        }
        p, div, span, label {
            color: #E3E3E3;
            font-size: 15px;
            line-height: 1.6;
        }
        .muted-text {
            color: #A8C7FA; /* Light blueish gray for secondary text */
            font-size: 13px;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #444746;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px;
            color: #C4C7C5;
            font-size: 16px;
            font-weight: 500;
            border: none;
            padding: 0 4px;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #A8C7FA; /* Google Blue */
            border-bottom: 3px solid #A8C7FA;
        }

        /* Cards & Containers */
        .notebook-card {
            background-color: #1E1F20;
            border: 1px solid #444746;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        
        /* Inputs */
        .stTextInput input, .stNumberInput input {
            background-color: #1E1F20 !important;
            border: 1px solid #444746 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            padding: 12px 16px !important;
        }
        .stTextInput input:focus {
            border-color: #A8C7FA !important;
            box-shadow: 0 0 0 1px #A8C7FA !important;
        }

        /* Buttons */
        div.stButton > button[kind="primary"] {
        background-color: #2A2B2E !important;   /* Dark grey */
        color: #EDEDED !important;              /* Light text */
        border: 1px solid #444746 !important;
        border-radius: 24px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }

    /* Hover state */
    div.stButton > button[kind="primary"]:hover {
        background-color: #3A3B3E !important;  /* Slightly lighter */
        border-color: #5F6368 !important;
        transform: translateY(-1px);
    }

    /* Active / pressed state */
    div.stButton > button[kind="primary"]:active {
        background-color: #1F2023 !important;  /* Slightly darker */
        transform: translateY(0);
    }

    /* Disabled state */
    div.stButton > button[kind="primary"]:disabled {
        background-color: #1E1F20 !important;
        color: #9AA0A6 !important;
        border-color: #444746 !important;
        cursor: not-allowed;
    }

            /* Primary FORM submit button (Ask button) */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #2A2B2E !important;   /* Dark grey */
            color: #EDEDED !important;
            border: 1px solid #444746 !important;
            border-radius: 24px;
            font-weight: 600;
            transition: all 0.15s ease-in-out;
        }

        /* Hover */
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #3A3B3E !important;
            border-color: #5F6368 !important;
            transform: translateY(-1px);
        }

        /* Active */
        div[data-testid="stFormSubmitButton"] > button:active {
            background-color: #1F2023 !important;
        }

        /* Disabled */
        div[data-testid="stFormSubmitButton"] > button:disabled {
            background-color: #1E1F20 !important;
            color: #9AA0A6 !important;
            border-color: #444746 !important;
            cursor: not-allowed;
        }

                
        /* Delete Button (Red) */
        div.stButton > button[kind="secondary"] {
            background-color: transparent;
            color: #F2B8B5;
            border: 1px solid #F2B8B5;
        }
        div.stButton > button[kind="secondary"]:hover {
            background-color: rgba(242, 184, 181, 0.1);
            color: #F2B8B5;
        }

        /* File Uploader */
        div[data-testid="stFileUploader"] {
            background-color: #1E1F20;
            border: 1px dashed #444746;
            border-radius: 12px;
            padding: 32px;
        }

        /* Align chat input and Ask button horizontally */
            .chat-row {
                display: flex;
                align-items: center;
            }

            /* Remove extra margin from input wrapper */
            .chat-row .stTextInput {
                margin-bottom: 0 !important;
            }

            /* Align button vertically with input */
            .chat-row .stButton {
                margin-top: 0 !important;
            }


        /* Badges */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 500;
            background-color: #0F5223;
            color: #C4EED0;
            border: 1px solid #37BE5F;
            margin-right: 8px;
        }
        .grounded-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 500;
            background-color: #004A77;
            color: #C2E7FF;
            border: 1px solid #00639B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- State Management ---
if "query_result" not in st.session_state:
    st.session_state.query_result = None

    

# --- Inngest Client ---
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

# --- Helper Functions ---
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

def list_uploaded_pdfs():
    return sorted([f.name for f in UPLOADS_DIR.glob("*.pdf")])

def save_uploaded_pdf(file) -> Path:
    path = UPLOADS_DIR / file.name
    path.write_bytes(file.getbuffer())
    return path

def delete_pdf(filename: str):
    path = UPLOADS_DIR / filename
    if path.exists():
        path.unlink()
        return True
    return False

async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]

def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def wait_for_run_output(event_id: str):
    base = _inngest_api_base()
    url = f"{base}/events/{event_id}/runs"

    with st.status("Retrieving evidence and generating answer...", expanded=False) as status:
        while True:
            try:
                resp = requests.get(url).json()
                runs = resp.get("data", [])
                if runs:
                    run = runs[0]
                    run_status = run.get("status")
                    if run_status in ("Completed", "Succeeded", "Success"):
                        status.update(label="Answer Ready", state="complete", expanded=False)
                        return run.get("output") or {}
                    if run_status in ("Failed", "Cancelled"):
                        status.update(label="Failed", state="error")
                        st.error("Query failed.")
                        return None
            except Exception:
                pass
            time.sleep(0.8)

# --- Main Layout ---

st.title("ContextClerk")

st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;">
  <img src="data:image/png;base64,{ICON_TOP}" width="36"/>
  <h2 style="margin:0;">Chat with your PDFs</h2>
</div>
<p class="muted">Ask questions across all uploaded documents and get grounded answers.</p>
""", unsafe_allow_html=True)

# Tabs
tab_sources, tab_chat = st.tabs(["Sources", "Chat"])

# --- Tab 1: Sources ---
with tab_sources:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;">
                 <img src="data:image/png;base64,{ICON_UPLOAD}" width="20"/>
        <h3 style="margin:0;">Add Sources</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='muted-text'>Upload PDF documents to create your knowledge base.</p>", unsafe_allow_html=True)
    
    # Upload Area
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("Process Files", type="primary"):
            with st.spinner("Reading documents..."):
                for file in uploaded_files:
                    path = save_uploaded_pdf(file)
                    asyncio.run(send_rag_ingest_event(path))
                    time.sleep(0.2)
            st.success(f"Successfully added {len(uploaded_files)} document(s) to the notebook.")
            time.sleep(1)
            st.rerun()

    st.divider()
    
    # Document List
    st.markdown("### Your Sources")
    docs = list_uploaded_pdfs()
    
    if not docs:
        st.info("No sources added yet.")
    else:
        for doc in docs:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(f"""
                <div class="notebook-card" style="padding: 12px; display: flex; align-items: center; gap:12px;">
                    <img src="data:image/png;base64,{ICON_PDF}" width="20"/>
                    <span style="font-weight: 500;">{doc}</span>
                </div>
                """, unsafe_allow_html=True)

            with col_del:
                if st.button("delete", key=f"del_{doc}", help="Delete source"):
                    if delete_pdf(doc):
                        st.toast(f"Deleted {doc}")
                        time.sleep(0.5)
                        st.rerun()

# --- Tab 2: Chat ---
with tab_chat:
    st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;">
            <img src="data:image/png;base64,{ICON_CHAT}" width="18"/>
            <h3 style="margin:0;">Chat with your sources</h3>
        </div>
        """, unsafe_allow_html=True)

    
    if not list_uploaded_pdfs():
        st.warning("Please add at least one source in the 'Sources' tab to start chatting.")
    else:
        # Chat Input
        with st.form("chat_form", border=False):
            st.markdown('<div class="chat-row">', unsafe_allow_html=True)

            col_in, col_btn = st.columns([5, 1], gap="small")

            with col_in:
                question = st.text_input(
                    "Ask a question",
                    placeholder="What does the document say about...",
                    label_visibility="collapsed"
                )

            with col_btn:
                submit = st.form_submit_button(
                    "Ask",
                    type="primary",
                    use_container_width=True
                )

            st.markdown('</div>', unsafe_allow_html=True)



                
                
        if submit and question:
            event_id = asyncio.run(send_rag_query_event(question, top_k=5))
            st.session_state.query_result = wait_for_run_output(event_id)

        # Result Display
        res = st.session_state.query_result
        if res:
            answer = res.get("answer", "")
            confidence = res.get("confidence", 0.0)
            grounded = res.get("grounded", False)
            evidence = res.get("evidence", [])

            st.markdown("---")
            
            # Answer Card
            st.markdown(f"""
            <div class="notebook-card">
                <div style="font-size: 16px; line-height: 1.7; margin-bottom: 16px;">
                    {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-top:8px;">
                
                <div class="confidence-badge">
                    <img src="data:image/png;base64,{ICON_CONF}"
                        width="14"
                        style="vertical-align:middle;margin-right:6px;" />
                    Confidence: {confidence:.2f}
                </div>

                <div class="grounded-badge">
                    <img src="data:image/png;base64,{ICON_OK if grounded else ICON_WARN}"
                        width="12"
                        style="vertical-align:middle;margin-right:6px;" />
                    {"Grounded" if grounded else "Ungrounded"}
                </div>

                </div>
                """, unsafe_allow_html=True)





            # Citations 
            if evidence:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;">
                <img src="data:image/png;base64,{ICON_SRC}" width="18"/>
                <h4 style="margin:0;">Sources</h4>
                </div>
                """, unsafe_allow_html=True)


                for idx, e in enumerate(evidence):
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;font-weight:600;">
                    <img src="data:image/png;base64,{ICON_PDF}" width="16"/>
                    <span>{e['source']}</span>
                    </div>
                    <div style="font-size:13px;color:#A8C7FA;">Page {e['page']}</div>
                    """, unsafe_allow_html=True)
