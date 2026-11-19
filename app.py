"""
DriveWise — Metadata-Aware Automotive RAG Assistant (Streamlit App)
Dark modern theme + demo-safe retrieval/generation.
"""

import os
import json
import time
import hashlib
import sqlite3
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
from google import genai
from google.genai import types
from pypdf import PdfReader

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="DriveWise", page_icon="🚗", layout="wide")

# Plotly dark template so charts blend into the theme
pio.templates.default = "plotly_dark"

# ============================================================
# Custom CSS — dark modern theme
# ============================================================
st.markdown("""
<style>
/* User bubble — indigo tint */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 18px 18px 4px 18px;
    padding: 4px 8px;
    margin: 6px 0;
}
/* Assistant bubble — slate with left accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #1e293b;
    border-left: 3px solid #818cf8;
    border-radius: 4px 18px 18px 18px;
    padding: 4px 8px;
    margin: 6px 0;
}
/* Metric cards on the dashboard */
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 16px;
}
/* Expander styling */
[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
}
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e293b;
}
/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
}
/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DB_PATH = "logs/query_logs.db"
BROCHURE_DIR = "brochures"
SAMPLES_BASE = "https://raw.githubusercontent.com/avanishar/drivewiseapp/main"
SAMPLE_FILES = ["Mahindra_XUV700.pdf", "Honda_Amaze.pdf", "Hundai_Creta.pdf"]

SECTION_KEYWORDS = {
    "Engine & Performance": ["engine", "torque", "power", "gearbox", "transmission", "hp", "ps", "cc",
        "cylinder", "performance", "speed", "acceleration", "manual", "automatic", "dct", "cvt", "bhp", "rpm"],
    "Mileage & Fuel Efficiency": ["mileage", "fuel economy", "fuel efficiency", "kmpl", "km/l",
        "consumption", "hybrid", "electric range", "efficiency", "co2", "emissions", "arai", "wltp"],
    "Safety": ["safety", "airbag", "abs", "ebd", "esc", "brake", "crash test", "ncap", "adas",
        "lane assist", "isofix", "hill assist", "esp", "traction control", "rear view camera", "tpms"],
    "Dimensions": ["dimensions", "length", "width", "height", "wheelbase", "ground clearance",
        "boot space", "weight", "capacity", "turning radius", "fuel tank", "kerb weight", "mm"],
    "Interior & Comfort": ["interior", "comfort", "seat", "upholstery", "climate control", "ac",
        "sunroof", "steering", "cabin", "leather", "ventilated", "ambient lighting", "armrest", "cruise control"],
    "Infotainment & Connectivity": ["infotainment", "screen", "display", "apple carplay", "android auto",
        "bluetooth", "speakers", "audio", "navigation", "connected car", "usb", "voice command", "touchscreen"],
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "down", "during", "each", "for", "from", "had", "has", "have",
    "having", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "you", "your",
}

# ============================================================
# API key + Gemini client
# ============================================================
def get_api_key():
    key = None
    try:
        key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.environ.get("GOOGLE_API_KEY")
    return key


api_key = get_api_key()
if not api_key:
    st.error(
        "No `GOOGLE_API_KEY` found.\n\n"
        "- **Locally**: set it as an environment variable or create `.streamlit/secrets.toml`.\n"
        "- **On Streamlit Cloud**: add it under **Settings → Secrets**."
    )
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"


def generate_with_retry(prompt, system_instruction=None, max_retries=3):
    delay = 5
    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
            ) if system_instruction else None
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            err = str(e).lower()
            if any(t in err for t in ["429", "quota", "limit", "exhausted", "500", "503", "server"]):
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
            else:
                raise


def generate_stream(prompt, system_instruction=None):
    """Streaming with non-streaming fallback. Never raises a traceback to the UI."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    ) if system_instruction else None
    try:
        response = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as stream_err:
        print(f"[generate_stream] streaming failed: {stream_err}. Falling back.")
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            if resp.text:
                yield resp.text
            else:
                yield "The model returned an empty response. Please try rephrasing your question."
        except Exception as fallback_err:
            print(f"[generate_stream] fallback failed: {fallback_err}")
            yield (
                "I couldn't reach the language model right now (server error). "
                "Please try again in a moment."
            )


# ============================================================
# Session state
# ============================================================
if "index_data" not in st.session_state:
    st.session_state.index_data = {"files": {}, "chunks": []}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

index_data = st.session_state.index_data


# ============================================================
# Chunking + metadata extraction
# ============================================================
def classify_section(text):
    scores = {sec: 0 for sec in SECTION_KEYWORDS}
    tl = text.lower()
    for sec, kws in SECTION_KEYWORDS.items():
        for kw in kws:
            scores[sec] += tl.count(kw)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "General Specifications"


def extract_metadata(filepath, first_page_text):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    brand, model, version = "Unknown", name_without_ext.title(), "1.0"
    for sep in ["_", "-"]:
        if sep in name_without_ext:
            parts = name_without_ext.split(sep)
            brand = parts[0].strip().title()
            model = " ".join(parts[1:]).strip().title()
            break

    try:
        prompt = f"""
        Extract the car brand, model, and brochure document version from the following text
        of the first page of the car brochure. Look for indicators of document version such as
        a version number (e.g. v1.1, version 2.0), model year (e.g. MY24, MY2023), or
        publication month/year (e.g. 10/2023, July 2022).

        Text:
        ---
        {first_page_text[:2000]}
        ---
        Respond with ONLY a valid JSON object (no markdown blocks):
        {{"brand": "BrandName", "model": "ModelName", "version": "VersionInfo"}}
        If you cannot extract the version, return "1.0" as the default version.
        """
        response = generate_with_retry(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        brand_extracted = brand if brand != "Unknown" else data.get("brand", "").strip().title()
        model_extracted = model if brand != "Unknown" else data.get("model", "").strip().title()
        version_extracted = data.get("version", "1.0").strip()
        if not brand_extracted or brand_extracted.lower() == "unknown":
            brand_extracted = brand
        if not model_extracted or model_extracted.lower() == "unknown":
            model_extracted = model
        return brand_extracted, model_extracted, version_extracted
    except Exception:
        return brand, model, version


def chunk_pdf(filepath):
    reader = PdfReader(filepath)
    first_page_text = reader.pages[0].extract_text() if len(reader.pages) > 0 else ""
    brand, model, version = extract_metadata(filepath, first_page_text)
    chunks = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        paras = text.split("\n\n")
        current, current_len = [], 0
        for para in paras:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > 800 and current:
                chunk_text = "\n".join(current)
                chunks.append({"text": chunk_text, "brand": brand, "model": model, "version": version,
                    "section": classify_section(chunk_text), "page": page_idx + 1,
                    "source_file": os.path.basename(filepath)})
                current, current_len = [para], len(para)
            else:
                current.append(para)
                current_len += len(para)
        if current:
            chunk_text = "\n".join(current)
            chunks.append({"text": chunk_text, "brand": brand, "model": model, "version": version,
                "section": classify_section(chunk_text), "page": page_idx + 1,
                "source_file": os.path.basename(filepath)})
    return chunks, brand, model, version


def embed_chunks(chunks):
    BATCH = 15
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        contents = [
            f"Car Brand: {c['brand']}\nModel: {c['model']}\nVersion: {c['version']}\n"
            f"Section: {c['section']}\nPage: {c['page']}\n{c['text']}"
            for c in batch
        ]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(model=EMBED_MODEL, contents=contents)
                for j, emb in enumerate(resp.embeddings):
                    batch[j]["embedding"] = np.array(emb.values, dtype=np.float32)
                break
            except Exception as e:
                err_lower = str(e).lower()
                if any(t in err_lower for t in ["429", "500", "503", "connection", "timeout", "server"]) and attempt < 5:
                    time.sleep(5 * (2 ** attempt))
                else:
                    raise
        time.sleep(0.3)


def index_pdf(filepath):
    fname = os.path.basename(filepath)
    fhash = hashlib.md5(open(filepath, "rb").read()).hexdigest()
    if fname in index_data["files"] and index_data["files"][fname].get("hash") == fhash:
        return
    index_data["chunks"] = [c for c in index_data["chunks"] if c.get("source_file") != fname]
    chunks, brand, model, version = chunk_pdf(filepath)
    embed_chunks(chunks)
    index_data["chunks"].extend(chunks)
    index_data["files"][fname] = {"hash": fhash, "brand": brand, "model": model,
                                   "version": version, "chunks_count": len(chunks)}
    return brand, model, version, len(chunks)


def load_samples():
    os.makedirs(BROCHURE_DIR, exist_ok=True)
    for fname in SAMPLE_FILES:
        fpath = f"{BROCHURE_DIR}/{fname}"
        if not os.path.exists(fpath):
            try:
                req = urllib.request.Request(f"{SAMPLES_BASE}/{fname}", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(fpath, "wb") as out_file:
                    out_file.write(response.read())
            except Exception:
                pass
    try:
        with open("brochure_index.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.index_data.clear()
        st.session_state.index_data.update(data)
    except Exception as e:
        st.error(f"Failed to load pre-built index: {e}")


def get_car_map():
    cm = {}
    for meta in index_data["files"].values():
        b, m = meta.get("brand", "Unknown"), meta.get("model", "Unknown")
        cm.setdefault(b, [])
        if m not in cm[b]:
            cm[b].append(m)
    return cm


# ============================================================
# Retrieval + Re-ranking
# ============================================================
def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0


def kw_score(query, text):
    words = [w.strip("?,.:;!\"'()").lower() for w in query.split() if w.lower() not in STOPWORDS]
    if not words:
        return 0.0
    tl = text.lower()
    return sum(1 for w in words if w in tl) / len(words)


def rerank_chunks(query, chunks, limit=4):
    """Re-rank with the LLM. On any parse/failure, fall back to Stage-1 order
    (which is still query-dependent) and log the reason."""
    if not chunks or len(chunks) <= 1:
        return chunks[:limit]
    try:
        candidates_str = "".join(
            f"\n--- Candidate [{idx}] (Pg {c['page']}) ---\n{c['text']}\n" for idx, c in enumerate(chunks)
        )
        prompt = f"""
        You are an expert automotive search ranker. Re-rank the following candidate chunks
        from a car brochure based on their relevance to the user's query.

        User Query: "{query}"

        Candidate Chunks:
        {candidates_str}

        Re-rank these candidates from most relevant to least relevant. Return the ordered list
        of candidate indices as a JSON array of integers, e.g. [2, 0, 1].
        Do NOT explain your reasoning, do NOT output markdown code blocks, respond with ONLY the JSON array.
        """
        response = generate_with_retry(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        ordered_indices = json.loads(text.strip())
        ordered_indices = [int(i) for i in ordered_indices if 0 <= int(i) < len(chunks)]
        reranked = [chunks[i] for i in ordered_indices]
        for c in chunks:
            if c not in reranked:
                reranked.append(c)
        return reranked[:limit]
    except Exception as e:
        print(f"[rerank_chunks] failed ({e}); using Stage-1 order.")
        return chunks[:limit]


def retrieve(query, brand, model, limit=4, debug=False):
    """Hybrid retrieval + optional re-ranking. Returns (chunks, debug_info)."""
    debug_info = {"filtered_count": 0, "scored": [], "rerank_used": False, "error": None}
    filtered = [c for c in index_data["chunks"]
                if c.get("brand", "").lower() == brand.lower()
                and c.get("model", "").lower() == model.lower()]
    debug_info["filtered_count"] = len(filtered)
    if not filtered:
        return [], debug_info

    try:
        emb_resp = client.models.embed_content(model=EMBED_MODEL, contents=query)
        q_emb = np.array(emb_resp.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        debug_info["error"] = f"embedding failed: {e}"
        print(f"[retrieve] query embedding failed: {e}")
        return [], debug_info

    scored = []
    missing_emb = 0
    for c in filtered:
        emb = c.get("embedding")
        if emb is None:
            missing_emb += 1
            continue
        emb = np.array(emb, dtype=np.float32) if not isinstance(emb, np.ndarray) else emb
        sem = cosine_sim(q_emb, emb)
        kw = kw_score(query, c["text"])
        scored.append({**c, "score": 0.8 * sem + 0.2 * kw})
    debug_info["missing_embedding_chunks"] = missing_emb
    debug_info["scored"] = sorted(scored, key=lambda x: x["score"], reverse=True)[:6]

    if not scored:
        debug_info["error"] = "no chunks had embeddings"
        return [], debug_info

    scored.sort(key=lambda x: x["score"], reverse=True)
    candidates = scored[:limit * 2]

    skip_rerank = os.environ.get("DRIVEWISE_SKIP_RERANK", "0") == "1"
    if skip_rerank:
        debug_info["rerank_used"] = False
        return candidates[:limit], debug_info

    reranked = rerank_chunks(query, candidates, limit)
    debug_info["rerank_used"] = True
    return reranked, debug_info


def build_context(chunks):
    return "".join(
        f"\n--- Source [{i+1}] (Page {c['page']}, Section: {c['section']}, Version: {c.get('version', '1.0')}) ---\n{c['text']}\n"
        for i, c in enumerate(chunks)
    )


def get_system_prompt(brand, model):
    return (
        f"You are an expert automotive assistant for Drive Wise. "
        f"Answer ONLY from the brochure excerpts for {brand} {model}. "
        "Rules: 1) Use only the provided context. 2) If missing, say so. "
        "3) Use inline citations [1],[2] matching source numbers. 4) Be clear and concise."
    )


# ============================================================
# LLM-as-judge evaluation + persistent SQLite logging
# ============================================================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, query TEXT, brand TEXT, model TEXT, response TEXT,
            response_time REAL, is_failed INTEGER,
            faithfulness REAL, context_relevance REAL, answer_correctness REAL,
            retrieved_chunks TEXT
        )
    """)
    conn.commit()
    conn.close()


def evaluate_answer(query, chunks, answer):
    if not chunks or not answer:
        return 1.0, 1.0, 1.0, "Empty chunks or response."
    try:
        context_str = "".join(f"\nSource [{idx+1}]: {c['text']}\n" for idx, c in enumerate(chunks))
        eval_prompt = f"""
        You are a RAG quality evaluation judge. Evaluate the retrieval and generation quality.

        Query: "{query}"
        Retrieved Context Chunks:
        {context_str}
        Generated Answer: "{answer}"

        Score the following from 1.0 (worst) to 5.0 (best):
        1. Context Relevance: how relevant are the retrieved chunks to the query?
        2. Faithfulness: is the answer fully grounded in the retrieved chunks (1.0 if hallucinated, 5.0 if 100% grounded)?
        3. Answer Correctness & Completeness: does the answer fully and accurately resolve the query using only the context?

        Respond with ONLY a valid JSON object (no markdown blocks):
        {{"context_relevance": float, "faithfulness": float, "answer_correctness": float, "rationale": "string"}}
        """
        response = generate_with_retry(eval_prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        return (float(data.get("context_relevance", 1.0)), float(data.get("faithfulness", 1.0)),
                float(data.get("answer_correctness", 1.0)), data.get("rationale", ""))
    except Exception as e:
        return 1.0, 1.0, 1.0, f"Error: {e}"


def log_to_db(query, brand, model, answer, elapsed, is_failed, faithfulness, relevance, correctness, chunks):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    chunks_meta = [{"page": c["page"], "section": c["section"], "version": c.get("version", "1.0"),
                    "source_file": c["source_file"]} for c in chunks]
    conn.execute("""
        INSERT INTO query_logs (timestamp, query, brand, model, response, response_time,
            is_failed, faithfulness, context_relevance, answer_correctness, retrieved_chunks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), query, brand, model, answer, elapsed, int(is_failed),
          faithfulness, relevance, correctness, json.dumps(chunks_meta)))
    conn.commit()
    conn.close()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("🚗 DriveWise")
    st.caption("Metadata-Aware Automotive RAG Assistant")
    st.divider()
    st.subheader("📚 Brochure Library")

    if st.button("Load Sample Brochures", use_container_width=True):
        with st.spinner("Downloading & indexing sample brochures..."):
            load_samples()
        st.success("Sample brochures indexed!")

    uploaded_files = st.file_uploader("Upload brochure PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs(BROCHURE_DIR, exist_ok=True)
        for f in uploaded_files:
            path = os.path.join(BROCHURE_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            with st.status(f"Indexing {f.name}...", expanded=False) as status:
                st.write("Extracting text and metadata...")
                index_pdf(path)
                status.update(label=f"{f.name} indexed", state="complete")
        st.success(f"Indexed {len(uploaded_files)} file(s)!")

    st.divider()
    if index_data["files"]:
        st.caption(f"**{len(index_data['files'])} brochure(s) indexed:**")
        for meta in index_data["files"].values():
            st.caption(f"• {meta['brand']} {meta['model']} (v{meta.get('version', '1.0')}) — {meta['chunks_count']} chunks")
    else:
        st.caption("No brochures indexed yet.")

    st.divider()
    st.subheader("🛠️ Demo Controls")
    debug_mode = st.checkbox("Show retrieval debug info", value=False,
                             help="Print retrieved chunks + prompt for each query.")
    skip_rerank = st.checkbox("Skip LLM re-ranking (faster)", value=False,
                              help="Return Stage-1 hybrid order directly.")
    if skip_rerank:
        os.environ["DRIVEWISE_SKIP_RERANK"] = "1"
    else:
        os.environ["DRIVEWISE_SKIP_RERANK"] = "0"


# ============================================================
# Main — Chat + Dashboard
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
            padding: 28px 32px; border-radius: 16px; margin-bottom: 20px;
            border: 1px solid #334155;">
  <h1 style="margin: 0; font-size: 2.4rem; font-weight: 800;
             background: linear-gradient(90deg, #c7d2fe, #818cf8);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    🚗 Drive Wise
  </h1>
  <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1rem;">
    Metadata-Aware Automotive RAG Assistant
  </p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Quality Dashboard"])

with tab_chat:
    car_map = get_car_map()
    brands = sorted(car_map.keys())

    if not brands:
        st.info("No brochures indexed yet. Use the sidebar to load samples or upload PDFs.")
    else:
        col1, col2 = st.columns(2)
        brand = col1.selectbox("Brand", brands)
        model = col2.selectbox("Model", car_map.get(brand, []))
        car_key = f"{brand} {model}"
        st.session_state.chat_history.setdefault(car_key, [])

        # Display history
        for item in st.session_state.chat_history[car_key]:
            with st.chat_message("user"):
                st.write(item["query"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                if item.get("sources"):
                    with st.expander("📎 View Sources"):
                        for i, s in enumerate(item["sources"]):
                            st.caption(f"[{i+1}] Page {s['page']} · {s['section']} · v{s.get('version', '1.0')}")
                            st.text(s["text"][:300] + "..." if len(s["text"]) > 300 else s["text"])
                if item.get("metrics"):
                    m = item["metrics"]
                    st.caption(
                        f"📊 Faithfulness **{m['faithfulness']:.1f}/5** · "
                        f"Relevance **{m['relevance']:.1f}/5** · "
                        f"Correctness **{m['correctness']:.1f}/5** · {item['time']}s"
                    )

        query = st.chat_input(f"Ask about the {brand} {model}...")
        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.status("Processing your question...", expanded=True) as status:
                    st.write("🔍 Retrieving relevant brochure chunks...")
                    t0 = time.time()
                    chunks, dbg = retrieve(query, brand, model, debug=debug_mode)

                    if debug_mode:
                        st.write(f"**Debug:** {dbg['filtered_count']} chunks for this car, "
                                 f"missing embeddings: {dbg.get('missing_embedding_chunks', 0)}, "
                                 f"rerank used: {dbg['rerank_used']}")
                        if dbg.get("error"):
                            st.write(f"**Debug error:** {dbg['error']}")
                        if dbg.get("scored"):
                            st.write("**Top Stage-1 scores:**")
                            for s in dbg["scored"]:
                                st.write(f"- page {s['page']} | {s['section']} | score {s['score']:.3f} | {s['text'][:80]}...")

                    if not chunks:
                        st.write("❌ No relevant chunks found.")
                        status.update(label="No data found", state="error", expanded=False)
                        answer = f"No brochure data found for '{brand} {model}'."
                        st.write(answer)
                        sources = []
                        elapsed = time.time() - t0
                        relevance = faithfulness = correctness = 1.0
                    else:
                        st.write("📄 Building context...")
                        context_str = build_context(chunks)
                        sys_prompt = get_system_prompt(brand, model)
                        prompt = (f"Brochure Context for {brand} {model}:\n{context_str}\n"
                                  f"User Query: \"{query}\"\nGrounded Answer:")

                        if debug_mode:
                            st.write("**Debug: final prompt (first 1200 chars):**")
                            st.code(prompt[:1200])

                        st.write("✍️ Generating answer...")
                        answer = st.write_stream(generate_stream(prompt, system_instruction=sys_prompt))
                        elapsed = round(time.time() - t0, 2)

                        st.write("📊 Evaluating quality...")
                        relevance, faithfulness, correctness, rationale = evaluate_answer(query, chunks, answer)
                        sources = chunks
                        status.update(label=f"Completed in {elapsed}s", state="complete", expanded=False)

                if sources:
                    with st.expander("📎 View Sources"):
                        for i, s in enumerate(sources):
                            st.caption(f"[{i+1}] Page {s['page']} · {s['section']} · v{s.get('version', '1.0')}")
                            st.text(s["text"][:300] + "..." if len(s["text"]) > 300 else s["text"])

                is_failed = (not sources) or any(w in answer.lower() for w in ["sorry", "not available", "error"])
                st.caption(
                    f"📊 Faithfulness **{faithfulness:.1f}/5** · "
                    f"Relevance **{relevance:.1f}/5** · "
                    f"Correctness **{correctness:.1f}/5** · {elapsed}s"
                )

            log_to_db(query, brand, model, answer, elapsed, is_failed,
                      faithfulness, relevance, correctness, sources)
            st.session_state.chat_history[car_key].append({
                "query": query, "answer": answer, "sources": sources, "time": elapsed,
                "metrics": {"relevance": relevance, "faithfulness": faithfulness, "correctness": correctness}
            })

with tab_dashboard:
    st.subheader("Evaluation & Quality Monitoring")
    if not os.path.exists(DB_PATH):
        st.info("No query log database found yet. Ask a few questions in the Chat tab first!")
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM query_logs", conn)
        conn.close()

        if df.empty:
            st.info("Query log is empty. Ask a few questions in the Chat tab to populate this dashboard.")
        else:
            total_queries = len(df)
            avg_latency = df["response_time"].mean()
            failure_rate = (df["is_failed"].sum() / total_queries) * 100.0
            avg_quality = (df["faithfulness"].mean() + df["context_relevance"].mean() + df["answer_correctness"].mean()) / 3

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Queries", total_queries)
            c2.metric("Avg Latency", f"{avg_latency:.2f}s")
            c3.metric("Failure Rate", f"{failure_rate:.1f}%")
            c4.metric("Avg Quality Score", f"{avg_quality:.2f}/5.0")

            colA, colB = st.columns(2)
            with colA:
                st.caption("Average RAG Quality (LLM-as-judge)")
                quality_df = pd.DataFrame({
                    "Metric": ["Faithfulness", "Context Relevance", "Answer Correctness"],
                    "Score": [df["faithfulness"].mean(), df["context_relevance"].mean(), df["answer_correctness"].mean()],
                })
                fig1 = px.bar(quality_df, x="Metric", y="Score", color="Metric",
                              color_discrete_sequence=["#818cf8", "#34d399", "#fbbf24"],
                              text_auto=".2f")
                fig1.update_layout(showlegend=False, yaxis_range=[0, 5.5], height=350,
                                   paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                                   font_color="#e2e8f0")
                st.plotly_chart(fig1, use_container_width=True)
            with colB:
                st.caption("Quality Score Trend Over Time")
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["quality_avg"] = (df["faithfulness"] + df["context_relevance"] + df["answer_correctness"]) / 3
                trend = df.sort_values("timestamp").tail(20)
                fig2 = px.line(trend, x="timestamp", y="quality_avg", markers=True,
                               labels={"timestamp": "Time", "quality_avg": "Avg Quality"})
                fig2.update_layout(height=350, yaxis_range=[0, 5.5],
                                   paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                                   font_color="#e2e8f0")
                st.plotly_chart(fig2, use_container_width=True)

            st.caption("Recent log entries")
            st.dataframe(
                df[["timestamp", "brand", "model", "query", "response_time",
                    "is_failed", "faithfulness", "context_relevance", "answer_correctness"]].tail(10),
                use_container_width=True,
            )
