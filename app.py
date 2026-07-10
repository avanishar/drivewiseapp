"""
DriveWise — Metadata-Aware Automotive RAG Assistant (Streamlit App)

Standalone web version of the DriveWise notebook pipeline:
PDF chunking -> metadata extraction (brand/model/version) -> Gemini embeddings
-> hybrid retrieval + LLM re-ranking -> grounded generation -> LLM-as-judge
evaluation -> persistent SQLite logging -> quality dashboard.

Run locally:      streamlit run app.py
Deploy:           push this file + requirements.txt to GitHub, then deploy on
                  Streamlit Community Cloud (share.streamlit.io), and add your
                  GOOGLE_API_KEY under the app's Settings -> Secrets.
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
import google.generativeai as genai
from pypdf import PdfReader

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="DriveWise — Car Brochure Assistant", page_icon="🚗", layout="wide")

# Custom premium styling CSS (Premium Light Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Main container background - Clean Slate Light Theme */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #f8fafc 0%, #f1f5f9 90%);
        color: #1e293b;
    }
    
    /* Custom header design */
    .header-container {
        padding: 1.5rem 0rem 2rem 0rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #eff6ff 0%, rgba(255,255,255,0) 100%);
        border-radius: 12px;
        padding-left: 20px;
    }
    
    .header-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 300;
    }
    
    /* Card containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(29, 78, 216, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(29, 78, 216, 0.4) !important;
    }
    
    /* Sidebar adjustments */
    .stSelectbox label {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
# API key + Gemini setup
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
        "- **Locally**: set it as an environment variable, or create `.streamlit/secrets.toml` "
        "with `GOOGLE_API_KEY = \"your-key\"`.\n"
        "- **On Streamlit Cloud**: add it under your app's **Settings → Secrets**."
    )
    st.stop()

genai.configure(api_key=api_key)


def generate_content_with_retry(model, prompt, generation_config=None, max_retries=5):
    delay = 10
    for attempt in range(max_retries):
        try:
            if generation_config:
                return model.generate_content(prompt, generation_config=generation_config)
            return model.generate_content(prompt)
        except Exception as e:
            err_str = str(e).lower()
            if any(t in err_str for t in ["429", "quota", "limit", "exhausted"]):
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
            else:
                raise e


# ============================================================
# Session state (acts as the in-memory index + chat log)
# ============================================================
if "index_data" not in st.session_state:
    st.session_state.index_data = {"files": {}, "chunks": []}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # keyed by "Brand Model"

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
        Respond with ONLY a valid JSON object in this format (no markdown blocks, just raw JSON text):
        {{"brand": "BrandName", "model": "ModelName", "version": "VersionInfo"}}
        If you cannot extract the version, return "1.0" as the default version.
        """
        model_gen = genai.GenerativeModel("models/gemini-flash-latest")
        response = generate_content_with_retry(model_gen, prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        data = json.loads(text)
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
    TRANSIENT = ("429", "500", "503", "connection", "timeout")
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        contents = [
            f"Car Brand: {c['brand']}\nModel: {c['model']}\nVersion: {c['version']}\n"
            f"Section: {c['section']}\nPage: {c['page']}\n{c['text']}"
            for c in batch
        ]
        for attempt in range(6):
            try:
                resp = genai.embed_content(model="models/gemini-embedding-001", content=contents)
                for j, emb in enumerate(resp["embedding"]):
                    batch[j]["embedding"] = np.array(emb, dtype=np.float32)
                break
            except Exception as e:
                err_lower = str(e).lower()
                if any(t in err_lower for t in TRANSIENT) and attempt < 5:
                    time.sleep(5 * (2 ** attempt))
                else:
                    raise
        time.sleep(0.5)


def index_pdf(filepath):
    fname = os.path.basename(filepath)
    fhash = hashlib.md5(open(filepath, "rb").read()).hexdigest()
    if fname in index_data["files"] and index_data["files"][fname].get("hash") == fhash:
        return  # already indexed, unchanged
    index_data["chunks"] = [c for c in index_data["chunks"] if c.get("source_file") != fname]
    chunks, brand, model, version = chunk_pdf(filepath)
    embed_chunks(chunks)
    index_data["chunks"].extend(chunks)
    index_data["files"][fname] = {"hash": fhash, "brand": brand, "model": model,
                                   "version": version, "chunks_count": len(chunks)}
    return brand, model, version, len(chunks)


def load_samples():
    os.makedirs(BROCHURE_DIR, exist_ok=True)
    # Download PDFs in background so they exist in library
    for fname in SAMPLE_FILES:
        fpath = f"{BROCHURE_DIR}/{fname}"
        if not os.path.exists(fpath):
            try:
                req = urllib.request.Request(f"{SAMPLES_BASE}/{fname}", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(fpath, "wb") as out_file:
                    out_file.write(response.read())
            except Exception:
                pass
    # Read pre-built chunks index locally from disk (cloned by git)
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
# Retrieval + Re-ranking + Generation
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
        model_gen = genai.GenerativeModel("models/gemini-flash-latest")
        response = generate_content_with_retry(model_gen, prompt)
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
    except Exception:
        return chunks[:limit]


def retrieve(query, brand, model, limit=4):
    filtered = [c for c in index_data["chunks"]
                if c.get("brand", "").lower() == brand.lower()
                and c.get("model", "").lower() == model.lower()]
    if not filtered:
        return []
    emb_resp = genai.embed_content(model="models/gemini-embedding-001", content=query)
    q_emb = emb_resp["embedding"]
    scored = []
    for c in filtered:
        emb = c.get("embedding")
        if emb is None:
            continue
        sem = cosine_sim(q_emb, emb)
        kw = kw_score(query, c["text"])
        scored.append({**c, "score": 0.8 * sem + 0.2 * kw})
    scored.sort(key=lambda x: x["score"], reverse=True)
    candidates = scored[:limit * 2]
    return rerank_chunks(query, candidates, limit)


def generate_answer(query, brand, model):
    chunks = retrieve(query, brand, model)
    if not chunks:
        return f"No brochure data found for '{brand} {model}'.", []
    context_str = "".join(
        f"\n--- Source [{i+1}] (Page {c['page']}, Section: {c['section']}, Version: {c.get('version', '1.0')}) ---\n{c['text']}\n"
        for i, c in enumerate(chunks)
    )
    sys_prompt = (
        f"You are an expert automotive assistant for Drive Wise. "
        f"Answer ONLY from the brochure excerpts for {brand} {model}. "
        "Rules: 1) Use only the provided context. 2) If missing, say so. "
        "3) Use inline citations [1],[2] matching source numbers. 4) Be clear and concise."
    )
    prompt = f"Brochure Context for {brand} {model}:\n{context_str}\nUser Query: \"{query}\"\nGrounded Answer:"
    try:
        model_gen = genai.GenerativeModel(model_name="models/gemini-flash-latest", system_instruction=sys_prompt)
        resp = generate_content_with_retry(model_gen, prompt,
                                            generation_config=genai.types.GenerationConfig(temperature=0.1))
        return resp.text.strip(), chunks
    except Exception as e:
        return f"Error: {e}", []


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
        model_eval = genai.GenerativeModel("models/gemini-flash-latest")
        response = generate_content_with_retry(model_eval, eval_prompt)
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
# Sidebar — brochure library management
# ============================================================
with st.sidebar:
    st.header("🚗 DriveWise")
    st.caption("Metadata-Aware Automotive RAG Assistant")
    st.divider()
    st.subheader("📚 Brochure Library")

    if st.button("Load Sample Brochures", width="stretch"):
        with st.spinner("Downloading & indexing sample brochures (~2 min)..."):
            load_samples()
        st.success("Sample brochures indexed!")

    uploaded_files = st.file_uploader("Upload brochure PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs(BROCHURE_DIR, exist_ok=True)
        with st.spinner("Indexing uploaded brochure(s)..."):
            for f in uploaded_files:
                path = os.path.join(BROCHURE_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                index_pdf(path)
        st.success(f"Indexed {len(uploaded_files)} file(s)!")

    st.divider()
    if index_data["files"]:
        st.caption(f"**{len(index_data['files'])} brochure(s) indexed:**")
        for meta in index_data["files"].values():
            st.caption(f"• {meta['brand']} {meta['model']} (v{meta.get('version', '1.0')}) — {meta['chunks_count']} chunks")
    else:
        st.caption("No brochures indexed yet.")



# ============================================================
# Main area — Chat + Dashboard tabs
# ============================================================
st.markdown("""
    <div class="header-container">
        <div class="header-title">Drive Wise</div>
        <div class="header-subtitle">Metadata-Aware Automotive RAG Assistant — Guided Car Decisions</div>
    </div>
""", unsafe_allow_html=True)

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Quality Dashboard"])

with tab_chat:
    car_map = get_car_map()
    brands = sorted(car_map.keys())

    if not brands:
        st.info("No brochures indexed yet. Use the sidebar to load sample brochures or upload your own PDF.")
    else:
        col1, col2 = st.columns(2)
        brand = col1.selectbox("Brand", brands)
        model = col2.selectbox("Model", car_map.get(brand, []))
        car_key = f"{brand} {model}"
        st.session_state.chat_history.setdefault(car_key, [])

        for item in st.session_state.chat_history[car_key]:
            with st.chat_message("user"):
                st.write(item["query"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                if item.get("sources"):
                    tags = "  ".join(
                        f"`[{i+1}] Pg {s['page']} · {s['section']} · v{s.get('version', '1.0')}`"
                        for i, s in enumerate(item["sources"])
                    )
                    st.caption("Sources: " + tags)
                if item.get("metrics"):
                    m = item["metrics"]
                    st.caption(
                        f"📊 Faithfulness **{m['faithfulness']:.1f}/5** · "
                        f"Context Relevance **{m['relevance']:.1f}/5** · "
                        f"Answer Correctness **{m['correctness']:.1f}/5** · {item['time']}s"
                    )

        query = st.chat_input(f"Ask about the {brand} {model}...")
        if query:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving, re-ranking, and generating..."):
                    t0 = time.time()
                    answer, sources = generate_answer(query, brand, model)
                    elapsed = round(time.time() - t0, 2)
                    is_failed = (not sources) or any(w in answer.lower() for w in ["sorry", "not available", "error"])
                with st.spinner("Evaluating answer quality..."):
                    relevance, faithfulness, correctness, rationale = evaluate_answer(query, sources, answer)
                log_to_db(query, brand, model, answer, elapsed, is_failed,
                          faithfulness, relevance, correctness, sources)

                st.write(answer)
                if sources:
                    tags = "  ".join(
                        f"`[{i+1}] Pg {s['page']} · {s['section']} · v{s.get('version', '1.0')}`"
                        for i, s in enumerate(sources)
                    )
                    st.caption("Sources: " + tags)
                st.caption(
                    f"📊 Faithfulness **{faithfulness:.1f}/5** · Context Relevance **{relevance:.1f}/5** · "
                    f"Answer Correctness **{correctness:.1f}/5** · {elapsed}s"
                )

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
                st.bar_chart(quality_df, x="Metric", y="Score")
            with colB:
                st.caption("Query Distribution by Car Model")
                counts = df.groupby(["brand", "model"]).size().reset_index(name="Queries")
                counts["Car"] = counts["brand"] + " " + counts["model"]
                st.bar_chart(counts, x="Car", y="Queries")

            st.caption("Recent log entries")
            st.dataframe(
                df[["timestamp", "brand", "model", "query", "response_time",
                    "is_failed", "faithfulness", "context_relevance", "answer_correctness"]].tail(10),
                width="stretch",
            )
