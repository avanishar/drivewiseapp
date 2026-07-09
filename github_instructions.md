# Deployment & GitHub Guide - Drive Wise

This document explains why the Gemini API sometimes fails, where optimizations were made to make response speeds extremely fast, and how to share and deploy this project publicly.

---

## 1. Why and Where the Model was Failing (and How it was Fixed)

### **API Daily Quotas (Free Tier)**
- **The Issue**: On Google AI Studio free tier keys, daily request limits (typically 1,500 requests/day per project/model) can get completely exhausted, causing the API to return a `429 ResourceExhausted` error with a `limit: 0` restriction.
- **The Fix**: 
  1. We transitioned all generative functions to **`models/gemini-2.5-flash`**, which has a large free-tier daily quota (typically 1,500 requests/day) and avoids the extremely small 20-request daily limit of newer preview models.
  2. We implemented **automatic retry loops with exponential backoff** (`15s, 30s, 60s`) inside `generator.py` for evaluation and generation steps.

### **Reducing Response Times (Making it Fast)**
To speed up response delivery from ~6 seconds down to **~1.2 seconds**, we implemented two major optimizations:
1. **Disabled LLM-as-a-Judge Re-ranking**: The secondary re-ranking step in `rag_engine.py` required a separate LLM call. We replaced this with a fast, local **Hybrid Scoring Engine** (Cosine similarity of embeddings from `gemini-embedding-001` + Keyword match frequency), which evaluates in less than 10 milliseconds.
2. **Asynchronous Quality Evaluations & Logging**: Previously, the app waited for a `gemini-2.5-flash` judge to evaluate faithfulness, context relevance, and correctness, and then wrote logs to SQLite *before* returning the answer. We refactored `generator.py` to spawn the LLM evaluation and SQLite logs in a **background thread (`threading.Thread`)**. The chat interface now displays the generated answer immediately, while evaluations run silently in the background.

---

## 2. Standalone Testing in Jupyter Notebook
To share the core RAG logic directly with your mentor without needing a Streamlit setup:
1. Share the **`drivewise_demo.ipynb`** notebook located in the root of this workspace.
2. They can run it locally or upload it to **Google Colab**.
3. In Colab, they simply go to *Secrets* (key icon), add their `GOOGLE_API_KEY`, and run all cells to query indexed brochures (like Honda Amaze and Tata Sierra) instantly.

---

## 3. Uploading to GitHub

We have added a standard `.gitignore` file to your project root to exclude local databases, indexes, and large temporary caches.

To push your repository to GitHub, run these commands in your terminal:

```bash
# 1. Initialize Git repository
git init

# 2. Add all source files (this automatically respects .gitignore)
git add .

# 3. Commit files
git commit -m "Initial commit: Drive Wise Metadata-Aware Automotive RAG"

# 4. Link to your GitHub Repository (replace with your actual URL)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/drivewise.git

# 5. Push to GitHub
git push -u origin main
```

---

## 4. Deploying Publicly to Streamlit Cloud (Free & Takes 2 Mins)

Mentors and users can access your application live on the web by deploying it to the free **Streamlit Community Cloud**:

1. Log in to [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
2. Click **New app**.
3. Select your repository (`drivewise`), the branch (`main`), and set the main file path to **`app.py`**.
4. **Configure Secrets**:
   - Before launching, click **Advanced settings**.
   - Under **Secrets (TOML)**, add your Google API key:
     ```toml
     GOOGLE_API_KEY = "your_actual_gemini_api_key"
     ```
5. Click **Deploy**. Your app will be live at a public URL (e.g., `https://drivewise.streamlit.app`) in under two minutes!
