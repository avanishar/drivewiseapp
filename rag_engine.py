import os
import json
import time
import numpy as np
import google.generativeai as genai

# Setup Google Generative AI
genai.configure()

# Use __file__-based absolute paths so the app works on any server (including Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index", "brochure_index.json")

# Stopwords for simple keyword matching
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", 
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", 
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", 
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
    "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", 
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", 
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", 
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", 
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", 
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", 
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", 
    "yourselves"
}

# In-memory index cache — loaded once per server session, never re-read from disk
_INDEX_CACHE = None

def load_index(force_reload=False):
    """Loads the brochure index from disk into memory (cached after first load)."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None and not force_reload:
        return _INDEX_CACHE
    if not os.path.exists(INDEX_PATH):
        print(f"Index file not found at: {INDEX_PATH}")
        return None
    try:
        print(f"Loading index from disk: {INDEX_PATH}")
        with open(INDEX_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # Convert embedding lists back to numpy arrays for vector math
        for chunk in raw.get('chunks', []):
            if 'embedding' in chunk and chunk['embedding'] is not None:
                chunk['embedding'] = np.array(chunk['embedding'], dtype=np.float32)
        _INDEX_CACHE = raw
        print(f"Index loaded: {len(raw.get('chunks', []))} chunks from {len(raw.get('files', {}))} files.")
        return _INDEX_CACHE
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def clear_index_cache():
    """Clears the in-memory cache so the next query reloads from disk."""
    global _INDEX_CACHE
    _INDEX_CACHE = None
    print("Index cache cleared.")


def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def calculate_keyword_score(query, text):
    """
    Computes a basic overlap score for terms in the query and chunk text,
    normalizing by the query term count. Useful for exact match terms like numbers.
    """
    query_words = [w.strip("?,.:;!\"'()").lower() for w in query.split()]
    query_words = [w for w in query_words if w and w not in STOPWORDS]
    if not query_words:
        return 0.0
        
    text_lower = text.lower()
    matches = 0
    for word in query_words:
        if word in text_lower:
            matches += 1
            
    return matches / len(query_words)

def get_available_cars(index_data=None):
    """Returns a list of unique brands and their models present in the index."""
    if index_data is None:
        index_data = load_index()
    if not index_data:
        return {}
        
    cars = {}
    for filename, meta in index_data.get("files", {}).items():
        brand = meta.get("brand", "Unknown")
        model = meta.get("model", "Unknown")
        if brand not in cars:
            cars[brand] = []
        if model not in cars[brand]:
            cars[brand].append(model)
            
    # Also scan chunks in case files metadata is missing/incomplete
    for chunk in index_data.get("chunks", []):
        brand = chunk.get("brand")
        model = chunk.get("model")
        if brand and model:
            if brand not in cars:
                cars[brand] = []
            if model not in cars[brand]:
                cars[brand].append(model)
                
    # Sort models
    for brand in cars:
        cars[brand] = sorted(list(set(cars[brand])))
        
    return cars

def retrieve_chunks(query, brand, model, limit=4, target_section=None, index_data=None):
    """
    Retrieves and re-ranks the most relevant chunks for a specific brand and model.
    Accepts an optional pre-loaded index_data to avoid disk reads on every call.
    """
    if index_data is None:
        index_data = load_index()
    if not index_data:
        print("No index database found.")
        return []
        
    chunks = index_data.get("chunks", [])
    
    # 1. Metadata Filtering: Filter by selected Brand and Model
    filtered_chunks = [
        c for c in chunks 
        if c.get("brand", "").lower() == brand.lower() and c.get("model", "").lower() == model.lower()
    ]
    
    if not filtered_chunks:
        print(f"No chunks found for car: {brand} {model}")
        return []
        
    # Optional filtering by a selected brochure section
    if target_section and target_section != "All Sections":
        filtered_chunks = [c for c in filtered_chunks if c.get("section") == target_section]
        
    if not filtered_chunks:
        return []
        
    # 2. Vector Embed Query
    try:
        response = genai.embed_content(
            model='models/gemini-embedding-001',
            content=query
        )
        query_embedding = response['embedding']
    except Exception as e:
        print(f"Failed to embed query: {e}")
        return []
        
    # 3. Retrieve & Compute Hybrid Scores
    retrieved = []
    for chunk in filtered_chunks:
        emb = chunk.get("embedding")
        if emb is None:
            continue
            
        semantic_score = cosine_similarity(query_embedding, emb)
        keyword_score = calculate_keyword_score(query, chunk["text"])
        
        # Hybrid Scoring Heuristic
        # Give higher weight to semantic similarity but add a keyword boost for exact spec matching
        hybrid_score = 0.8 * semantic_score + 0.2 * keyword_score
        
        retrieved.append({
            "text": chunk["text"],
            "brand": chunk["brand"],
            "model": chunk["model"],
            "section": chunk["section"],
            "page": chunk["page"],
            "source_file": chunk["source_file"],
            "score": float(hybrid_score),
            "semantic_score": float(semantic_score),
            "keyword_score": float(keyword_score)
        })
        
    # Sort by hybrid score in descending order
    retrieved.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top candidates for re-ranking
    candidates = retrieved[:limit*2]
    
    # 4. Return top-K candidates directly for speed (avoiding slow LLM re-ranking call)
    return candidates[:limit]

if __name__ == "__main__":
    # Test retrieval
    cars = get_available_cars()
    print("Available cars:", cars)
