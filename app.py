"""
app.py  —  SQL RAG Query Generator  (Streamlit UI)
Run:  streamlit run app.py
"""

import os
import time
import traceback
from pathlib import Path

import sqlparse
import streamlit as st
from dotenv import load_dotenv

from sql_rag_engine import SQL_MODEL_ID, SQLRAGEngine

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SQL Query Generator",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
:root {
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --accent:#58a6ff; --accent2:#3fb950; --text:#e6edf3;
    --muted:#8b949e; --sql-bg:#0d1f0d;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Space Grotesk',sans-serif}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:'JetBrains Mono',monospace;border-radius:6px!important}
.stButton>button{background:var(--accent)!important;color:#0d1117!important;font-family:'Space Grotesk',sans-serif;font-weight:700;border:none!important;border-radius:6px!important;padding:.5rem 1.5rem!important}
.sql-output{background:var(--sql-bg);border:1px solid var(--accent2);border-left:4px solid var(--accent2);border-radius:8px;padding:1.2rem 1.5rem;font-family:'JetBrains Mono',monospace;font-size:.9rem;color:#a5f3a5;white-space:pre-wrap;line-height:1.7}
.schema-block{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--muted);white-space:pre-wrap;max-height:320px;overflow-y:auto}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem 1rem;text-align:center}
.metric-card .label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.metric-card .value{font-size:1.4rem;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace}
hr.c{border:none;border-top:1px solid var(--border);margin:1rem 0}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("engine", None), ("index_ready", False), ("history", []), ("last_result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ SQL Query Generator")
    st.markdown('<hr class="c">', unsafe_allow_html=True)

    hf_token = st.text_input("HuggingFace API Token", value=os.getenv("HF_TOKEN", ""),
                              type="password", placeholder="hf_xxxxxxxxxxxx")

    schema_path = st.text_input("Schema file path", value="schema.sql",
                                 placeholder="path/to/schema.sql")

    MODEL_OPTIONS = {
        "mistralai/Mistral-7B-Instruct-v0.3  (default)": "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-Coder-7B-Instruct  (best for SQL)": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct  (lightweight)": "microsoft/Phi-3.5-mini-instruct",
        "meta-llama/Llama-3.1-8B-Instruct  (strong)": "meta-llama/Llama-3.1-8B-Instruct",
    }
    selected_model = MODEL_OPTIONS[st.selectbox("Model", list(MODEL_OPTIONS.keys()))]
    top_k = st.slider("Schema chunks (top-k)", 2, 10, 6)

    st.markdown('<hr class="c">', unsafe_allow_html=True)

    if st.button("⚡ Build / Rebuild Index", use_container_width=True):
        if not hf_token:
            st.error("Enter your HuggingFace API token.")
        elif not Path(schema_path).exists():
            st.error(f"File not found: `{schema_path}`")
        else:
            with st.spinner("Parsing schema & building index…"):
                try:
                    engine = SQLRAGEngine(
                        schema_path=schema_path,
                        hf_token=hf_token,
                        model_id=selected_model,
                        top_k=top_k,
                    )
                    engine.build_index()
                    st.session_state.engine      = engine
                    st.session_state.index_ready = True
                    st.success("Index built successfully!")
                except Exception as e:
                    st.error(f"Index error: {e}")
                    st.code(traceback.format_exc())

    # Status badge
    if st.session_state.index_ready:
        st.markdown('<div class="metric-card"><div class="label">Index</div><div class="value" style="color:#3fb950">● Ready</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="label">Index</div><div class="value" style="color:#d29922">○ Not Built</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="c">', unsafe_allow_html=True)
    st.markdown("**💡 Example queries**")
    EXAMPLES = [
        "Give me top 5 entries with most number of protocol entries",
        "Show all critical alerts with device names and IP addresses",
        "Total bytes sent and received per device, ordered by total traffic",
        "List users and number of devices they own where count > 1",
        "All sessions longer than 10 seconds that used HTTPS protocol",
    ]
    for ex in EXAMPLES:
        label = ex[:52] + "…" if len(ex) > 55 else ex
        if st.button(f"› {label}", key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown('<hr class="c">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem;color:#8b949e;text-align:center">LlamaIndex · RAG · HuggingFace</div>', unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:JetBrains Mono;font-size:1.7rem;margin-bottom:.2rem'>"
    "🛢️ Natural Language → SQL</h1>"
    "<p style='color:#8b949e;margin-top:0'>RAG-powered · schema-aware · multi-table joins</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="c">', unsafe_allow_html=True)

prefill = st.session_state.pop("prefill", "")
user_question = st.text_area(
    "question", value=prefill, height=90,
    placeholder='e.g. "Give me top 5 entries with most number of protocol entries"',
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    generate_btn = st.button("▶ Generate SQL", use_container_width=True)
with col_clear:
    if st.button("✕ Clear"):
        st.session_state.last_result = None
        st.rerun()

# ── Generate ───────────────────────────────────────────────────────────────────
if generate_btn:
    if not st.session_state.index_ready:
        st.warning("⚠️  Build the index first (sidebar button).")
    elif not user_question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("🤖 Querying model…"):
            t0 = time.time()
            try:
                result = st.session_state.engine.generate(user_question.strip())
                result["elapsed"]  = round(time.time() - t0, 2)
                result["question"] = user_question.strip()
                st.session_state.last_result = result
                st.session_state.history.insert(0, result)
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop()
            except Exception as e:
                st.error(f"**Generation error:** {e or type(e).__name__}")
                with st.expander("🔍 Full traceback"):
                    st.code(traceback.format_exc(), language="python")

# ── Result ─────────────────────────────────────────────────────────────────────
if st.session_state.last_result:
    r = st.session_state.last_result
    st.markdown('<hr class="c">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="label">Time</div><div class="value">{r.get("elapsed","—")}s</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="label">Model</div><div class="value" style="font-size:.75rem">{r["model"].split("/")[-1]}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="label">SQL Lines</div><div class="value">{r["sql"].count(chr(10))+1}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Generated SQL")
    fmt = sqlparse.format(r["sql"], reindent=True, keyword_case="upper", indent_width=4)
    st.markdown(f'<div class="sql-output">{fmt}</div>', unsafe_allow_html=True)
    st.code(fmt, language="sql")

    with st.expander("📋 Schema context used"):
        st.markdown(f'<div class="schema-block">{r["schema_used"]}</div>', unsafe_allow_html=True)
    with st.expander("🔬 Full prompt"):
        st.markdown(f'<div class="schema-block">{r["prompt"]}</div>', unsafe_allow_html=True)

# ── History ─────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr class="c">', unsafe_allow_html=True)
    st.markdown("#### 🕐 Query History")
    for i, h in enumerate(st.session_state.history):
        label = h["question"][:80] + ("…" if len(h["question"]) > 80 else "")
        with st.expander(f"{i+1}.  {label}"):
            st.code(sqlparse.format(h["sql"], reindent=True, keyword_case="upper", indent_width=4), language="sql")