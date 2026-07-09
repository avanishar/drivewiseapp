import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join("logs", "query_logs.db")

def init_db():
    """Initializes the SQLite database if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            brand TEXT,
            model TEXT,
            response TEXT,
            response_time REAL,
            is_failed INTEGER,
            faithfulness REAL,
            context_relevance REAL,
            answer_correctness REAL,
            retrieved_chunks TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(query, brand, model, response, response_time, is_failed, faithfulness, context_relevance, answer_correctness, retrieved_chunks):
    """
    Logs a query and its metadata to the SQLite database.
    
    retrieved_chunks: A list of dicts containing source info.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    chunks_json = json.dumps(retrieved_chunks)
    
    cursor.execute("""
        INSERT INTO query_logs (
            timestamp, query, brand, model, response, response_time, 
            is_failed, faithfulness, context_relevance, answer_correctness, retrieved_chunks
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, query, brand, model, response, response_time,
        int(is_failed), faithfulness, context_relevance, answer_correctness, chunks_json
    ))
    conn.commit()
    conn.close()

def get_logs(limit=100):
    """Retrieves the latest logs from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for r in rows:
        log_entry = dict(r)
        # Parse chunks JSON
        try:
            log_entry['retrieved_chunks'] = json.loads(log_entry['retrieved_chunks'])
        except Exception:
            log_entry['retrieved_chunks'] = []
        logs.append(log_entry)
    return logs

def get_stats():
    """Retrieves aggregate statistics for monitoring."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total queries
    cursor.execute("SELECT COUNT(*) FROM query_logs")
    total_queries = cursor.fetchone()[0]
    
    if total_queries == 0:
        conn.close()
        return {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "failure_rate": 0.0,
            "avg_faithfulness": 0.0,
            "avg_context_relevance": 0.0,
            "avg_answer_correctness": 0.0,
            "queries_by_car": {}
        }
        
    # Avg response time
    cursor.execute("SELECT AVG(response_time) FROM query_logs")
    avg_response_time = cursor.fetchone()[0] or 0.0
    
    # Failure rate
    cursor.execute("SELECT SUM(is_failed) FROM query_logs")
    total_failed = cursor.fetchone()[0] or 0
    failure_rate = (total_failed / total_queries) * 100.0
    
    # Average evaluations (only average if not NULL)
    cursor.execute("SELECT AVG(faithfulness) FROM query_logs WHERE faithfulness IS NOT NULL")
    avg_faithfulness = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(context_relevance) FROM query_logs WHERE context_relevance IS NOT NULL")
    avg_context_relevance = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(answer_correctness) FROM query_logs WHERE answer_correctness IS NOT NULL")
    avg_answer_correctness = cursor.fetchone()[0] or 0.0
    
    # Queries by car
    cursor.execute("SELECT brand, model, COUNT(*) FROM query_logs GROUP BY brand, model")
    car_counts = cursor.fetchall()
    queries_by_car = {f"{c[0]} {c[1]}": c[2] for c in car_counts}
    
    conn.close()
    
    return {
        "total_queries": total_queries,
        "avg_response_time": round(avg_response_time, 2),
        "failure_rate": round(failure_rate, 2),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_context_relevance": round(avg_context_relevance, 2),
        "avg_answer_correctness": round(avg_answer_correctness, 2),
        "queries_by_car": queries_by_car
    }
