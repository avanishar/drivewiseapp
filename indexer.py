import os
import json
import time
import hashlib
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai

# Setup Google Generative AI — reads GOOGLE_API_KEY from environment (set as secret on HF Spaces)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Standard brochure sections
SECTIONS = [
    "Engine & Performance",
    "Mileage & Fuel Efficiency",
    "Safety",
    "Dimensions",
    "Interior & Comfort",
    "Infotainment & Connectivity",
    "General Specifications"
]

# Keyword mappings for rule-based section classification
SECTION_KEYWORDS = {
    "Engine & Performance": [
        "engine", "torque", "power", "gearbox", "transmission", "hp", "ps", "cc", "cylinder", 
        "performance", "speed", "acceleration", "manual", "automatic", "dct", "cvt", "imt", "bhp", "rpm"
    ],
    "Mileage & Fuel Efficiency": [
        "mileage", "fuel economy", "fuel efficiency", "kmpl", "km/l", "consumption", "hybrid", 
        "electric range", "efficiency", "co2", "emissions", "battery capacity", "range", "wltp", "arai"
    ],
    "Safety": [
        "safety", "airbag", "abs", "ebd", "esc", "brake", "crash test", "ncap", "adas", 
        "lane assist", "isofix", "hill assist", "esp", "traction control", "rear view camera", 
        "parking sensors", "child safety", "tpms"
    ],
    "Dimensions": [
        "dimensions", "length", "width", "height", "wheelbase", "ground clearance", "boot space", 
        "weight", "kg", "capacity", "turning radius", "fuel tank capacity", "kerb weight", "gross weight", "mm"
    ],
    "Interior & Comfort": [
        "interior", "comfort", "seat", "upholstery", "climate control", "ac", "sunroof", "steering", 
        "cabin", "leather", "ventilated", "glovebox", "charging", "ambient lighting", "armrest", 
        "headrest", "rear ac vent", "keyless", "cruise control"
    ],
    "Infotainment & Connectivity": [
        "infotainment", "screen", "display", "apple carplay", "android auto", "bluetooth", "speakers", 
        "audio", "navigation", "connected car", "app", "usb", "voice command", "touchscreen", "tweeters",
        "sound system", "ota updates"
    ]
}

# Use __file__-based absolute paths so the app works on any server (including Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index", "brochure_index.json")

def get_file_hash(filepath):
    """Generates an MD5 hash of the file to check if it has changed."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_brand_and_model(filepath, first_page_text):
    """
    Identifies the brand and model from the filename, or falls back to Gemini LLM 
    to extract them from the first page text.
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try parsing brand_model or brand-model
    for separator in ["_", "-"]:
        if separator in name_without_ext:
            parts = name_without_ext.split(separator)
            if len(parts) >= 2:
                brand = parts[0].strip().title()
                model = " ".join(parts[1:]).strip().title()
                return brand, model
                
    # Fallback to Gemini LLM
    try:
        prompt = f"""
        Extract the car brand and model from the following text of the first page of the car brochure.
        Text:
        ---
        {first_page_text[:1500]}
        ---
        Respond with ONLY a valid JSON object in this format (no markdown code blocks, just raw JSON text):
        {{"brand": "BrandName", "model": "ModelName"}}
        If you cannot extract the brand or model, guess based on common car brands (e.g., Hyundai, Tata, Suzuki, Honda, Toyota).
        """
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        import json
        text = response.text.strip()
        # Clean any accidental markdown format from response
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        data = json.loads(text)
        return data["brand"].strip().title(), data["model"].strip().title()
    except Exception as e:
        print(f"Error calling LLM for brand/model extraction: {e}")
        # Default fallback to filename
        return "Unknown Brand", name_without_ext.title()

def classify_section(text):
    """Classifies a chunk of text into one of the standard brochure sections."""
    scores = {sec: 0 for sec in SECTIONS[:-1]}
    text_lower = text.lower()
    
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            # Add score proportional to keyword count
            count = text_lower.count(kw)
            if count > 0:
                scores[section] += count
                
    # Get highest scoring section
    max_score = 0
    best_section = "General Specifications"
    for section, score in scores.items():
        if score > max_score:
            max_score = score
            best_section = section
            
    # If the score is too low, default to General Specifications
    if max_score < 2:
        return "General Specifications"
        
    return best_section

def get_chunks_from_pdf(filepath):
    """
    Reads PDF page by page, splits pages into paragraphs (structured chunking),
    classifies sections, and yields chunks with metadata.
    """
    reader = PdfReader(filepath)
    num_pages = len(reader.pages)
    
    # Extract first page text for metadata fallback identification
    first_page_text = ""
    if num_pages > 0:
        first_page_text = reader.pages[0].extract_text() or ""
        
    brand, model = extract_brand_and_model(filepath, first_page_text)
    print(f"Indexing brochure for: {brand} {model} (from {os.path.basename(filepath)})")
    
    chunks = []
    for page_idx in range(num_pages):
        page = reader.pages[page_idx]
        page_num = page_idx + 1
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue
            
        # Segment page into paragraphs/logical blocks (chunking strategy)
        paragraphs = page_text.split("\n\n")
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If the paragraph itself is huge, split it by lines
            if len(para) > 1000:
                sub_lines = para.split("\n")
                for line in sub_lines:
                    line = line.strip()
                    if not line:
                        continue
                    if current_len + len(line) > 800:
                        chunk_text = "\n".join(current_chunk)
                        section = classify_section(chunk_text)
                        chunks.append({
                            "text": chunk_text,
                            "brand": brand,
                            "model": model,
                            "section": section,
                            "page": page_num,
                            "source_file": os.path.basename(filepath)
                        })
                        current_chunk = [line]
                        current_len = len(line)
                    else:
                        current_chunk.append(line)
                        current_len += len(line)
            else:
                if current_len + len(para) > 800:
                    chunk_text = "\n".join(current_chunk)
                    section = classify_section(chunk_text)
                    chunks.append({
                        "text": chunk_text,
                        "brand": brand,
                        "model": model,
                        "section": section,
                        "page": page_num,
                        "source_file": os.path.basename(filepath)
                    })
                    current_chunk = [para]
                    current_len = len(para)
                else:
                    current_chunk.append(para)
                    current_len += len(para)
                    
        # Add residual chunk from the page
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            section = classify_section(chunk_text)
            chunks.append({
                "text": chunk_text,
                "brand": brand,
                "model": model,
                "section": section,
                "page": page_num,
                "source_file": os.path.basename(filepath)
            })
            
    return chunks

def build_index(brochures_dir=None):
    """
    Scans brochures_dir for PDFs, parses them, embeds the chunks, and updates the vector database.
    """
    if brochures_dir is None:
        brochures_dir = os.path.join(BASE_DIR, "brochures")
    os.makedirs(brochures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    # Load existing index if it exists
    existing_index = {}
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Convert embedding lists back to numpy arrays
            for chunk in raw.get('chunks', []):
                if 'embedding' in chunk and chunk['embedding'] is not None:
                    chunk['embedding'] = np.array(chunk['embedding'], dtype=np.float32)
            existing_index = raw
        except Exception as e:
            print(f"Error loading existing index, building new one: {e}")
            existing_index = {}
            
    # Form of existing_index:
    # {
    #     "files": { "filename.pdf": { "hash": "md5hash", "chunks_count": 12 } },
    #     "chunks": [ { "text": ..., "brand": ..., "model": ..., "section": ..., "page": ..., "embedding": ..., "source_file": ... } ]
    # }
    
    files_metadata = existing_index.get("files", {})
    all_chunks = existing_index.get("chunks", [])
    
    pdf_files = [f for f in os.listdir(brochures_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF brochures found in 'brochures/'. Please add some.")
        return False
        
    index_updated = False
    
    for filename in pdf_files:
        filepath = os.path.join(brochures_dir, filename)
        file_hash = get_file_hash(filepath)
        
        # Check if file has already been indexed and is unchanged
        if filename in files_metadata and files_metadata[filename].get("hash") == file_hash:
            print(f"Skipping {filename} - already indexed and unchanged.")
            continue
            
        index_updated = True
        
        # Remove any old chunks from this file if we are re-indexing it
        if filename in files_metadata:
            print(f"Re-indexing {filename} due to changes...")
            all_chunks = [c for c in all_chunks if c.get("source_file") != filename]
            
        # Parse chunks
        try:
            new_chunks = get_chunks_from_pdf(filepath)
            
            # Embed chunks
            print(f"Generating embeddings for {len(new_chunks)} chunks using models/gemini-embedding-001...")
            
            # Batch embedding requests to be faster and respect limits
            # We will embed chunks in batches of 20
            batch_size = 20
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i+batch_size]
                
                # Format each chunk for the embedding model to incorporate metadata context
                contents = []
                for c in batch:
                    formatted_content = f"Car Brand: {c['brand']}\nCar Model: {c['model']}\nSection: {c['section']}\nPage: {c['page']}\nDetails: {c['text']}"
                    contents.append(formatted_content)
                
                # API Call with robust rate limit handling
                response = None
                max_retries = 6
                current_delay = 8
                for attempt in range(max_retries):
                    try:
                        response = genai.embed_content(
                            model='models/gemini-embedding-001',
                            content=contents
                        )
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
                            print(f"Rate limit hit during embedding. Sleeping for {current_delay}s (Attempt {attempt+1}/{max_retries})...")
                            time.sleep(current_delay)
                            current_delay *= 2
                        else:
                            raise e
                
                if response is None:
                    raise RuntimeError("Failed to generate embeddings after multiple retries due to API quota limits.")
                
                # Assign embeddings back to chunks
                embeddings = response['embedding']
                for chunk_idx, emb in enumerate(embeddings):
                    batch[chunk_idx]['embedding'] = emb
                
                # Sleep to stagger requests
                time.sleep(0.5)
                
            all_chunks.extend(new_chunks)
            files_metadata[filename] = {
                "hash": file_hash,
                "chunks_count": len(new_chunks),
                "indexed_at": time.time(),
                "brand": new_chunks[0]["brand"] if new_chunks else "Unknown",
                "model": new_chunks[0]["model"] if new_chunks else "Unknown"
            }
            print(f"Successfully indexed {filename} ({len(new_chunks)} chunks).")
            
        except Exception as e:
            print(f"Failed to index {filename}: {e}")
            import traceback
            traceback.print_exc()
            
    if index_updated:
        # Convert numpy embeddings to plain Python lists for cross-platform JSON storage
        serializable_chunks = []
        for chunk in all_chunks:
            c = dict(chunk)
            if 'embedding' in c and c['embedding'] is not None:
                emb = c['embedding']
                c['embedding'] = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            serializable_chunks.append(c)
        # Save as JSON (works on any OS / Python version)
        with open(INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump({"files": files_metadata, "chunks": serializable_chunks}, f)
        print("Brochure index successfully saved to disk as JSON.")
        return True
    else:
        print("All files are up to date. Index not modified.")
        return False

if __name__ == "__main__":
    build_index()
