import os
import streamlit as st
import pandas as pd
from datetime import datetime
import indexer
import rag_engine
import generator
import logger

# Set page config - MUST be the very first Streamlit call
st.set_page_config(
    page_title="Drive Wise - Metadata Aware Automotive RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-rebuild index on first launch if no cars are found (important for cloud deployments)
# This ensures Streamlit Cloud always has a working index even if the .pkl was built on a different OS.
if "index_initialized" not in st.session_state:
    st.session_state["index_initialized"] = True
    cars_check = rag_engine.get_available_cars()
    if not cars_check:
        with st.spinner("🔄 First launch: Building brochure index... (this takes ~2 minutes)"):
            indexer.build_index()
        st.success("✅ Index built successfully! Refreshing...")
        st.rerun()


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
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .user-bubble {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        color: #1e293b;
    }
    
    .assistant-bubble {
        background: #ffffff;
        border-left: 4px solid #10b981;
        color: #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Source panel */
    .source-tag {
        display: inline-block;
        background: #ccfbf1;
        color: #0d9488;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        margin-right: 0.5rem;
        border: 1px solid #99f6e4;
    }
    
    .source-file-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
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

# Main Title Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">Drive Wise</div>
        <div class="header-subtitle">Metadata-Aware Automotive RAG Assistant — Guided Car Decisions</div>
    </div>
""", unsafe_allow_html=True)

# Fetch current cars database
cars_db = rag_engine.get_available_cars()

# Sidebar: Controls & Selection
with st.sidebar:
    st.markdown("### 🚗 Configuration")
    
    if cars_db:
        # User select brand
        brands = sorted(list(cars_db.keys()))
        selected_brand = st.selectbox("Select Brand", brands)
        
        # User select model
        models = cars_db[selected_brand]
        selected_model = st.selectbox("Select Model", models)
        
        st.markdown(f"Currently querying: **{selected_brand} {selected_model}**")
    else:
        st.warning("No brochures indexed yet.")
        selected_brand = None
        selected_model = None

    st.markdown("---")
    st.markdown("### 📁 Document Management")
    st.info("Place new PDF brochures in the `brochures/` folder and click below to process.")
    
    if st.button("🔄 Rebuild / Index Brochures"):
        with st.spinner("Processing brochures... (Extracting text, chunking, and generating embeddings)"):
            updated = indexer.build_index()
            if updated:
                st.success("Index updated successfully!")
                time_sleep = 1
                st.rerun()
            else:
                st.info("All brochures are already up to date.")
                
    st.markdown("---")
    st.markdown("### 💡 Active Sections Filter")
    sections_list = ["All Sections", "Engine & Performance", "Mileage & Fuel Efficiency", "Safety", "Dimensions", "Interior & Comfort", "Infotainment & Connectivity", "General Specifications"]
    selected_section = st.selectbox("Search only in:", sections_list)

# Tabs
tab1, tab2 = st.tabs(["💬 Assistant", "📊 Analytics & Quality Tracking"])

# TAB 1: Chat Assistant
with tab1:
    if not selected_brand or not selected_model:
        st.markdown("""
            <div style="background: rgba(255, 165, 0, 0.1); border-left: 4px solid orange; padding: 1.5rem; border-radius: 8px;">
                <h4>Welcome to Drive Wise!</h4>
                <p>No processed brochures were found in the index database. Let's get started:</p>
                <ol>
                    <li>Copy your car brochures (.pdf) into the <b><code>brochures/</code></b> directory inside the workspace.</li>
                    <li>Click the <b>"Rebuild / Index Brochures"</b> button in the sidebar.</li>
                    <li>Choose a Brand and Model to start asking questions!</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Chat Interface
        st.markdown(f"#### Ask any question about the **{selected_brand} {selected_model}**")
        
        # Initialize session state for chat history
        chat_key = f"chat_{selected_brand}_{selected_model}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
            
        # Display chat history
        for message in st.session_state[chat_key]:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"""
                    <div class="chat-bubble user-bubble">
                        <b>👤 You:</b><br>{content}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-bubble assistant-bubble">
                        <b>🤖 Drive Wise:</b><br>{content}
                    </div>
                """, unsafe_allow_html=True)
                
                # Show source attribution details
                if "sources" in message and message["sources"]:
                    with st.expander("🔍 Citations & Sources"):
                        for src in message["sources"]:
                            st.markdown(f"""
                                <div style="margin-bottom: 0.75rem; padding: 0.5rem; background: rgba(255,255,255,0.01); border-radius: 6px;">
                                    <span class="source-tag">[{src['citation_num']}] Section: {src['section']}</span>
                                    <span class="source-file-tag">File: {src['source_file']} (Page {src['page']})</span>
                                    <p style="font-size: 0.9rem; margin-top: 0.4rem; color: #cbd5e0; line-height: 1.5; font-style: italic;">
                                        "{src['text']}"
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)

        # Chat Input Form
        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your question (e.g., 'What safety features are standard?' or 'What is the ground clearance?')")
            submit_button = st.form_submit_button("Send Query")
            
        if submit_button and user_query:
            # Show user message immediately
            st.markdown(f"""
                <div class="chat-bubble user-bubble">
                    <b>👤 You:</b><br>{user_query}
                </div>
            """, unsafe_allow_html=True)
            
            # Generate RAG response
            with st.spinner("Analyzing brochure details..."):
                rag_result = generator.generate_grounded_answer(
                    query=user_query,
                    brand=selected_brand,
                    model=selected_model,
                    target_section=selected_section if selected_section != "All Sections" else None
                )
                
            # Render answer
            st.markdown(f"""
                <div class="chat-bubble assistant-bubble">
                    <b>🤖 Drive Wise:</b><br>{rag_result['answer']}
                </div>
            """, unsafe_allow_html=True)
            
            # Render sources
            if rag_result["sources"]:
                with st.expander("🔍 Citations & Sources"):
                    for src in rag_result["sources"]:
                        st.markdown(f"""
                            <div style="margin-bottom: 0.75rem; padding: 0.5rem; background: rgba(255,255,255,0.01); border-radius: 6px;">
                                <span class="source-tag">[{src['citation_num']}] Section: {src['section']}</span>
                                <span class="source-file-tag">File: {src['source_file']} (Page {src['page']})</span>
                                <p style="font-size: 0.9rem; margin-top: 0.4rem; color: #cbd5e0; line-height: 1.5; font-style: italic;">
                                    "{src['text']}"
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Save to session history
            st.session_state[chat_key].append({"role": "user", "content": user_query})
            st.session_state[chat_key].append({
                "role": "assistant", 
                "content": rag_result["answer"],
                "sources": rag_result["sources"]
            })
            
            # Rerun to cleanly draw
            st.rerun()

# TAB 2: Analytics & Logs Dashboard
with tab2:
    st.markdown("### 📊 System Performance & RAG Metrics Dashboard")
    
    # Load aggregate statistics
    stats = logger.get_stats()
    
    # Display top level cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{stats['total_queries']}</div>
                <div class="metric-label">Total Queries Logged</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{stats['avg_response_time']}s</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{stats['failure_rate']}%</div>
                <div class="metric-label">Query Failure Rate</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        # Calculate combined quality index
        qual_avg = (stats['avg_faithfulness'] + stats['avg_context_relevance'] + stats['avg_answer_correctness']) / 3
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{round(qual_avg, 2)} / 5.0</div>
                <div class="metric-label">Average Quality Score</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Quality dimensions columns
    st.markdown("#### RAG Quality Metrics (LLM-as-a-Judge)")
    qcol1, qcol2, qcol3 = st.columns(3)
    
    with qcol1:
        st.metric(
            label="Faithfulness (Groundedness)", 
            value=f"{stats['avg_faithfulness']} / 5.0", 
            help="Measures if the generated answer is strictly supported by the retrieved contexts, without hallucinations."
        )
        
    with qcol2:
        st.metric(
            label="Context Relevance", 
            value=f"{stats['avg_context_relevance']} / 5.0", 
            help="Measures if the chunks retrieved from the PDF brochure are relevant and helpful for the user query."
        )
        
    with qcol3:
        st.metric(
            label="Answer Correctness & Completeness", 
            value=f"{stats['avg_answer_correctness']} / 5.0", 
            help="Measures if the answer correctly and fully answers the query using only the brochure details."
        )
        
    # Chart plotting query frequency and car selections
    st.markdown("---")
    ccol1, ccol2 = st.columns(2)
    
    logs_data = logger.get_logs(limit=200)
    
    with ccol1:
        st.markdown("#### Queries by Car Model")
        if stats['queries_by_car']:
            car_df = pd.DataFrame(list(stats['queries_by_car'].items()), columns=["Car Model", "Count"])
            st.bar_chart(car_df.set_index("Car Model"))
        else:
            st.info("No query logs available yet.")
            
    with ccol2:
        st.markdown("#### Latency & Evaluation Trend")
        if logs_data:
            df = pd.DataFrame(logs_data)
            # Normalize timestamp to readable time
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_sorted = df.sort_values('timestamp')
            trend_df = df_sorted[['timestamp', 'response_time', 'faithfulness', 'context_relevance']].dropna()
            if not trend_df.empty:
                st.line_chart(trend_df.set_index('timestamp')[['response_time', 'faithfulness', 'context_relevance']])
            else:
                st.info("Insufficient metrics logged for trend chart.")
        else:
            st.info("No query logs available yet.")
            
    # Recent log list table
    st.markdown("---")
    st.markdown("#### Recent Queries History")
    if logs_data:
        history_df = pd.DataFrame(logs_data)
        # Select and rename columns for clean user display
        display_columns = {
            "timestamp": "Time",
            "brand": "Brand",
            "model": "Model",
            "query": "User Query",
            "response_time": "Latency (s)",
            "is_failed": "Failed?",
            "faithfulness": "Faithfulness",
            "context_relevance": "Context Relevance",
            "answer_correctness": "Correctness"
        }
        
        # Keep only existing columns
        cols_to_use = [c for c in display_columns.keys() if c in history_df.columns]
        table_df = history_df[cols_to_use].rename(columns=display_columns)
        st.dataframe(table_df, use_container_width=True)
    else:
        st.info("No query logs available yet.")
