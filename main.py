"""RAG Assistant — Streamlit UI with glassmorphism + minimalism layering."""

import os

import streamlit as st

import rag_pipeline

st.set_page_config(
    page_title="Lumen RAG — Ask Your Documents",
    page_icon="✦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Glassmorphism + minimalism styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --glass: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.18);
        --stroke: rgba(255, 255, 255, 0.14);
        --accent: rgba(0, 212, 255, 0.9);
        --accent-soft: rgba(0, 212, 255, 0.18);
        --text-hi: #f4f8ff;
        --text-mid: rgba(235, 242, 255, 0.72);
        --text-low: rgba(235, 242, 255, 0.5);
    }

    /* Animated layered gradient background */
    .stApp {
        background:
            radial-gradient(1100px 700px at 12% -8%, rgba(99,102,241,0.28), transparent 60%),
            radial-gradient(1000px 760px at 108% -6%, rgba(0,212,255,0.20), transparent 58%),
            radial-gradient(900px 700px at 50% 120%, rgba(236,72,153,0.16), transparent 55%),
            linear-gradient(160deg, #0b1023 0%, #101632 45%, #0a0e1f 100%);
    }

    /* floating glass orbs */
    .stApp::before, .stApp::after {
        content: "";
        position: fixed;
        z-index: 0;
        border-radius: 50%;
        filter: blur(70px);
        opacity: 0.5;
        pointer-events: none;
    }
    .stApp::before {
        width: 380px; height: 380px;
        top: 12%; left: -120px;
        background: radial-gradient(circle, rgba(0,212,255,0.35), transparent 70%);
        animation: drift 22s ease-in-out infinite alternate;
    }
    .stApp::after {
        width: 460px; height: 460px;
        bottom: 4%; right: -140px;
        background: radial-gradient(circle, rgba(168,85,247,0.32), transparent 70%);
        animation: drift 26s ease-in-out infinite alternate-reverse;
    }
    @keyframes drift {
        from { transform: translateY(0) translateX(0); }
        to   { transform: translateY(40px) translateX(-30px); }
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* central column layering so content sits above orbs */
    [data-testid="stAppViewContainer"] > .main .block-container {
        position: relative;
        z-index: 2;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    /* Hide default chrome for a cleaner canvas */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer, [data-testid="stToolbar"] {
        visibility: hidden;
    }

    /* ---------- Sidebar: glass card ---------- */
    [data-testid="stSidebar"] {
        background: rgba(16, 20, 40, 0.35);
        backdrop-filter: blur(22px) saturate(160%);
        -webkit-backdrop-filter: blur(22px) saturate(160%);
        border-right: 1px solid var(--glass-border);
    }
    [data-testid="stSidebar"] > div {
        padding-top: 1.6rem;
    }
    .side-brand {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0 1rem 1.2rem;
    }
    .side-brand .logo {
        width: 38px; height: 38px; flex: 0 0 38px;
        border-radius: 12px;
        display: grid; place-items: center;
        font-size: 20px; color: #06122b;
        background: linear-gradient(135deg, #22d3ee, #818cf8);
        box-shadow: 0 0 24px rgba(34,211,238,0.45);
    }
    .side-brand .name { font-weight: 600; font-size: 1.05rem; letter-spacing: 0.3px; }
    .side-brand .sub { font-size: 0.7rem; color: var(--text-low); font-weight: 400; }

    /* glass panel */
    .glass {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        backdrop-filter: blur(18px) saturate(150%);
        -webkit-backdrop-filter: blur(18px) saturate(150%);
        box-shadow: 0 20px 50px -20px rgba(0,0,0,0.6);
        padding: 1.3rem 1.2rem;
        margin-bottom: 1.1rem;
    }
    .glass .g-title {
        display: flex; align-items: center; gap: 0.5rem;
        font-weight: 600; font-size: 0.95rem; letter-spacing: 0.4px;
        color: var(--text-hi); margin-bottom: 0.9rem;
    }

    /* uploader: transparent, fits glass panel */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px dashed var(--stroke) !important;
        border-radius: 14px !important;
        padding: 0.8rem !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* ---------- Main header ---------- */
    .hero {
        display: flex; align-items: center; gap: 1rem;
        margin-bottom: 1.6rem;
    }
    .hero .orb {
        width: 54px; height: 54px; flex: 0 0 54px; border-radius: 16px;
        display: grid; place-items: center; font-size: 28px;
        background: linear-gradient(135deg, #22d3ee, #818cf8);
        box-shadow: 0 10px 30px -8px rgba(129,140,248,0.6),
                    inset 0 0 0 1px rgba(255,255,255,0.25);
    }
    .hero h1 {
        font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px;
        margin: 0; color: var(--text-hi);
        background: linear-gradient(90deg, #eaf6ff, #a5b4fc);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero .tagline {
        color: var(--text-mid); font-size: 0.92rem; font-weight: 300;
        margin-top: 2px;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 12px 36px -22px rgba(0,0,0,0.7);
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        border-radius: 50%;
        background: linear-gradient(135deg, #22d3ee, #818cf8);
        color: #06122b;
    }

    /* chat input: floating glass pill */
    [data-testid="stChatInput"] {
        border: 1px solid var(--glass-border);
        border-radius: 999px;
        background: var(--glass);
        backdrop-filter: blur(18px) saturate(150%);
        -webkit-backdrop-filter: blur(18px) saturate(150%);
        box-shadow: 0 18px 44px -22px rgba(0,0,0,0.7);
    }
    [data-testid="stChatInput"] textarea { background: transparent; }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #22d3ee, #818cf8);
        border-radius: 999px; color: #06122b;
    }
    [data-testid="stChatInput"] button:hover {
        filter: brightness(1.1);
    }

    /* ---------- Buttons / file pills ---------- */
    .file-row {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.55rem 0.7rem;
        margin: 0.4rem 0;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--stroke);
    }
    .file-row .fname {
        flex: 1; font-size: 0.85rem; color: var(--text-mid);
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .del-link {
        font-size: 0.8rem; color: #fda4af; cursor: pointer;
        text-decoration: none; padding: 2px 8px; border-radius: 8px;
        border: 1px solid rgba(253,164,175,0.3);
    }
    .del-link:hover { background: rgba(244,63,94,0.15); }

    .muted { color: var(--text-low); font-size: 0.85rem; }

    /* metric chips */
    [data-testid="stMetric"] {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 0.7rem;
    }

    /* clean scrollbars */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.18); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.28); }

    /* subtle section label */
    .sec-label {
        font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase;
        color: var(--text-low); margin: 0.4rem 0 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — Document Manager (glass panel)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="side-brand">
        <div class="logo">✦</div>
        <div>
            <div class="name">Lumen RAG</div>
            <div class="sub">Document Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="g-title">📂 Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            save_path = os.path.join(
                str(rag_pipeline.DATA_DIR), uploaded.name
            )
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                with st.spinner(f"Indexing {uploaded.name}…"):
                    rag_pipeline.process_and_store_pdf(save_path)
                st.toast(f"✓ Indexed {uploaded.name}")

    st.markdown('<div class="sec-label">Active Documents</div>', unsafe_allow_html=True)
    current_files = rag_pipeline.get_all_uploaded_files()

    if current_files:
        for name in current_files:
            col1, col2 = st.columns([5, 1])
            col1.markdown(
                f'<div class="fname">📄 {name}</div>',
                unsafe_allow_html=True,
            )
            if col2.button("🗑", key=f"del_{name}", help=f"Delete {name}"):
                rag_pipeline.delete_file_and_embeddings(name)
                st.toast(f"Deleted {name}")
                st.rerun()
    else:
        st.markdown(
            '<div class="muted">No documents uploaded yet.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="muted" style="margin-top:0.8rem">🗂️ {len(current_files)} active</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar stats (layered minimalism)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="g-title">⚡ Model</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    colA.metric("Indexed chunks", rag_pipeline.get_vectorstore()._collection.count())
    colB.metric("Embeddings", rag_pipeline.EMBEDDING_MODEL.split("/")[-1])
    st.markdown(
        f'<div class="muted" style="margin-top:0.5rem">Qwen 72B · {rag_pipeline.TOP_K} excerpts</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main — Chat interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="orb">✦</div>
        <div>
            <h1>Ask your documents</h1>
            <div class="tagline">Conversational Q&A across your PDFs, with sources.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Conversation header hint (minimal)
if not st.session_state.messages:
    st.markdown(
        '<div class="muted" style="margin-bottom:0.8rem">Begin by uploading PDFs, '
        "then ask a question below.</div>",
        unsafe_allow_html=True,
    )

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                response = rag_pipeline.search_and_generate(prompt)
            except Exception as exc:  # surface pipeline errors gracefully
                response = f"⚠️ Something went wrong: {exc}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
