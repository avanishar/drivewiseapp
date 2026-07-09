import time
import json
import threading
import google.generativeai as genai
from rag_engine import retrieve_chunks
from logger import log_query

# Setup Google Generative AI
genai.configure()

def run_evaluation_and_log_async(query, brand, model, response_text, is_failed, elapsed_time, sources_list, context_str):
    """
    Evaluates response quality using LLM-as-a-judge and logs the query details 
    in a background thread to prevent UI blocking.
    """
    def run():
        faithfulness = 0.0
        context_relevance = 0.0
        answer_correctness = 0.0
        
        if not is_failed:
            try:
                eval_prompt = f"""
                You are a RAG quality evaluation judge. Your task is to evaluate the quality of the retrieval and generation system.
                
                Inputs to evaluate:
                - Query: "{query}"
                - Retrieved Context Chunks:
                {context_str}
                - Generated Answer: "{response_text}"
                
                Evaluate and score the following metrics on a scale from 1.0 (worst) to 5.0 (best):
                1. Context Relevance: How relevant and helpful are the retrieved chunks to the user's specific query?
                2. Faithfulness: Is the generated answer completely grounded in the retrieved chunks? Award a 1.0 if it makes claims not supported by the context or hallucinates. Award a 5.0 if it is 100% grounded.
                3. Answer Correctness & Completeness: Does the generated answer accurately and completely resolve the user query based strictly on the retrieved context?
                
                Respond with ONLY a valid JSON object in this format (no markdown blocks, just raw JSON text):
                {{
                  "context_relevance": float,
                  "faithfulness": float,
                  "answer_correctness": float,
                  "rationale": "string explanation of scores"
                }}
                """
                model_eval = genai.GenerativeModel("models/gemini-2.5-flash")
                
                # API Call with robust rate limit retries
                eval_response = None
                max_retries = 5
                delay = 15
                for attempt in range(max_retries):
                    try:
                        eval_response = model_eval.generate_content(eval_prompt)
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "exhausted" in err_str:
                            if attempt == max_retries - 1:
                                raise e
                            print(f"Rate limit hit during evaluation. Sleeping {delay}s (Attempt {attempt+1}/{max_retries})...")
                            time.sleep(delay)
                            delay *= 2
                        else:
                            raise e
                
                eval_text = eval_response.text.strip()
                if eval_text.startswith("```json"):
                    eval_text = eval_text[7:]
                if eval_text.endswith("```"):
                    eval_text = eval_text[:-3]
                eval_text = eval_text.strip()
                
                eval_data = json.loads(eval_text)
                context_relevance = float(eval_data.get("context_relevance", 0.0))
                faithfulness = float(eval_data.get("faithfulness", 0.0))
                answer_correctness = float(eval_data.get("answer_correctness", 0.0))
                
            except Exception as e:
                print(f"Error during LLM evaluation: {e}")
                
        # Write to local database log
        log_query(
            query=query,
            brand=brand,
            model=model,
            response=response_text,
            response_time=elapsed_time,
            is_failed=is_failed,
            faithfulness=faithfulness if not is_failed else None,
            context_relevance=context_relevance if not is_failed else None,
            answer_correctness=answer_correctness if not is_failed else None,
            retrieved_chunks=sources_list
        )
        print(f"Logged query: '{query}' with background evaluation results.")
        
    threading.Thread(target=run, daemon=True).start()

def generate_grounded_answer(query, brand, model, target_section=None):
    """
    Retrieves relevant brochure chunks, generates a grounded response with citations,
    spawns background quality evaluation, and returns immediately.
    """
    start_time = time.time()
    
    # 1. Retrieve relevant chunks (includes pre-filtering and speed-optimized local sort)
    chunks = retrieve_chunks(query, brand, model, limit=4, target_section=target_section)
    
    if not chunks:
        response_text = f"I couldn't find any brochure details for **{brand} {model}** in my index. Please make sure the brochure PDF is uploaded and processed."
        elapsed_time = time.time() - start_time
        # Log failure synchronously since there's no LLM load
        log_query(
            query=query, brand=brand, model=model, response=response_text,
            response_time=elapsed_time, is_failed=True,
            faithfulness=None, context_relevance=None, answer_correctness=None,
            retrieved_chunks=[]
        )
        return {
            "answer": response_text,
            "sources": [],
            "metrics": {
                "response_time": elapsed_time,
                "faithfulness": 0.0,
                "context_relevance": 0.0,
                "answer_correctness": 0.0
            }
        }
        
    # 2. Format Context for the Generator Model
    context_str = ""
    for idx, c in enumerate(chunks):
        context_str += f"\n--- Source [{idx+1}] (Page {c['page']}, Section: {c['section']}) ---\n{c['text']}\n"
        
    # 3. Generate Answer
    system_instruction = f"""
    You are an expert automotive assistant for the Drive Wise application. Your job is to answer the user's query about the selected car: {brand} {model}.
    
    You must follow these rules strictly:
    1. Ground your answer ONLY in the provided brochure excerpts under the "Brochure Context" section.
    2. Do NOT use any pre-existing knowledge about the car or assume specifications. If the information is not in the brochure context, state: "I'm sorry, but that information is not available in the brochure details for this vehicle."
    3. Include inline citations to the sources using numbers like [1], [2], etc., corresponding to the Source index listed in the context.
    4. Keep your answer professional, clear, structured, and easy for a car buyer to understand.
    """
    
    prompt = f"""
    Brochure Context for {brand} {model}:
    {context_str}
    
    User Query: "{query}"
    
    Grounded Answer (remember to use [1], [2] inline citations matching the source numbers above):
    """
    
    is_failed = False
    try:
        model_gen = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            system_instruction=system_instruction
        )
        # API Call with robust rate limit retries
        response = None
        max_retries = 5
        delay = 15
        for attempt in range(max_retries):
            try:
                response = model_gen.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.1)
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "exhausted" in err_str:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Rate limit hit during generation. Sleeping {delay}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
                    
        response_text = response.text.strip()
        
        # Check if the model failed to find the answer in the text
        lower_resp = response_text.lower()
        if "not available in the brochure" in lower_resp or "sorry" in lower_resp or "could not find" in lower_resp:
            is_failed = True
            
    except Exception as e:
        print(f"Error during response generation: {e}")
        response_text = "An error occurred while generating the answer. Please try again."
        is_failed = True
        
    elapsed_time = time.time() - start_time
    
    # Clean citations structure for frontend display
    sources_list = []
    for idx, c in enumerate(chunks):
        sources_list.append({
            "citation_num": idx + 1,
            "text": c["text"],
            "section": c["section"],
            "page": c["page"],
            "source_file": c["source_file"],
            "score": c.get("score", 0.0)
        })
        
    # 4. Spawn background evaluation and logging thread to prevent UI blocking
    run_evaluation_and_log_async(
        query=query,
        brand=brand,
        model=model,
        response_text=response_text,
        is_failed=is_failed,
        elapsed_time=elapsed_time,
        sources_list=sources_list,
        context_str=context_str
    )
    
    return {
        "answer": response_text,
        "sources": sources_list,
        "metrics": {
            "response_time": elapsed_time,
            "faithfulness": 0.0,  # Computed in background
            "context_relevance": 0.0,
            "answer_correctness": 0.0
        }
    }
