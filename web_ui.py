from flask import Flask, render_template, request, jsonify, stream_with_context, Response
import yaml
import sys
import os
import threading
import queue
import time
import json
import subprocess # Import subprocess for explicit process type check

# Add the parent directory of app.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the refactored backend logic
import app as app_backend

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this to a strong, random key!

# Global variables to hold initialized RAG components
rag_system_initialized = False
llm_instance = None
knowledge_bases = None
llamafile_process = None # To manage the subprocess object

# Queue for streaming LLM responses and initialization status to the frontend
response_queue = queue.Queue()

# --- Configuration Loading ---
CONFIG_FILE = 'config.yaml'
ui_config = {}

def load_config():
    """Loads UI configuration from config.yaml."""
    global ui_config
    try:
        with open(CONFIG_FILE, 'r') as f:
            ui_config = yaml.safe_load(f)
        print(f"[web_ui.py] Loaded configuration from {CONFIG_FILE}")
    except FileNotFoundError:
        print(f"[web_ui.py] Error: {CONFIG_FILE} not found. Using default empty config.")
        ui_config = {}
    except yaml.YAMLError as e:
        print(f"[web_ui.py] Error parsing {CONFIG_FILE}: {e}. Using default empty config.")
        ui_config = {}

load_config() # Load config at startup

# --- RAG System Initialization (runs once when Flask app starts) ---
def initialize_rag_system_thread():
    """Initializes the RAG system in a separate thread."""
    global rag_system_initialized, llm_instance, knowledge_bases, llamafile_process 
    print("[web_ui.py] Initializing RAG system (this may take a moment)...")
    try:
        # Pass ui_config to the backend initialization function
        llm_instance, knowledge_bases, llamafile_process = app_backend.initialize_rag_system(ui_config)
        
        if llm_instance and knowledge_bases:
            rag_system_initialized = True
            print("[web_ui.py] RAG system initialized successfully!")
            response_queue.put({"type": "init_complete", "success": True})
        else:
            print("[web_ui.py] RAG system initialization failed (backend returned None).")
            response_queue.put({"type": "init_complete", "success": False})
    except Exception as e:
        print(f"[web_ui.py] Critical error during RAG system initialization: {e}")
        response_queue.put({"type": "init_complete", "success": False, "error": str(e)})

# Start RAG system initialization in a background thread
init_thread = threading.Thread(target=initialize_rag_system_thread)
init_thread.daemon = True # Allow main program to exit even if this thread is running
init_thread.start()

# --- Helper for JSON serialization of Documents ---
def convert_docs_to_dicts(docs):
    """Converts a list of Langchain Document objects to a list of dictionaries for JSON serialization."""
    converted_docs = []
    for doc in docs:
        # Ensure page_content is a string and metadata is a dictionary
        content = str(doc.page_content) if doc.page_content is not None else ""
        metadata = dict(doc.metadata) if doc.metadata is not None else {}
        converted_docs.append({"page_content": content, "metadata": metadata})
    return converted_docs

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Pass UI configuration to the HTML template
    return render_template('index.html', ui_config=ui_config)

@app.route('/stream_init_status')
def stream_init_status():
    """Streams initialization status to the frontend."""
    def generate():
        while True:
            item = response_queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] == "init_complete":
                break
            time.sleep(0.1) 
    return Response(generate(), mimetype='text/event-stream')


@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries and streams LLM responses."""
    if not rag_system_initialized:
        return jsonify({"error": "RAG system not initialized. Please wait or check server logs."}), 503

    user_input = request.json.get('query', '').strip()
    if not user_input:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Clear the queue for a new response
    while not response_queue.empty():
        try:
            response_queue.get_nowait()
        except queue.Empty:
            pass

    # Process query in a separate thread to avoid blocking Flask
    def process_query_thread(input_query):
        global llm_instance, knowledge_bases
        query_results = {
            "llm_response": "",
            "referenced_articles": [], # List of dicts for direct_headlines
            "citations": [] # List of dicts for other RAG
        }
        
        try:
            selected_category, original_query_text, query_text_for_llm, metadata_filter = \
                app_backend.parse_user_input(input_query, knowledge_bases.keys())

            if selected_category == 'direct_headlines':
                # This returns a list of dicts: [{"title": "", "summary": "", ...}]
                retrieved_headlines_list = app_backend.get_headlines_from_filesystem(original_query_text, metadata_filter)
                
                if not retrieved_headlines_list:
                    query_results["llm_response"] = "No headlines found matching your criteria from the file system."
                else:
                    # Format for LLM context to generate executive summary
                    llm_context_for_executive_summary = []
                    for item in retrieved_headlines_list:
                        llm_context_for_executive_summary.append(
                            f"- Source: {item['source']}, Site: {item['site_origin']}, Title: {item['title']}, Summary: {item['summary']}"
                        )
                    context_str_for_llm = "\n".join(llm_context_for_executive_summary)

                    executive_summary_prompt = app_backend.EXECUTIVE_SUMMARY_PROMPT.format(context=context_str_for_llm)
                    
                    messages_for_llm = [
                        {"role": "system", "content": app_backend.SYSTEM_INSTRUCTION_PROMPT},
                        {"role": "user", "content": executive_summary_prompt}
                    ]

                    # Stream executive summary
                    for chunk in llm_instance.stream(messages_for_llm):
                        if chunk.content:
                            response_queue.put({"type": "llm_chunk", "content": chunk.content})
                    
                    query_results["referenced_articles"] = retrieved_headlines_list # Pass raw headlines for display

            elif selected_category in [app_backend.NEWS_SUMMARIES_KB_NAME, app_backend.BLOG_SUMMARIES_KB_NAME, app_backend.SOCIAL_MEDIA_SUMMARIES_KB_NAME]:
                # handle_summary_query now returns the LLM response and retrieved docs
                llm_response_generator, retrieved_docs_for_ui = app_backend.handle_summary_query(
                    llm_instance, knowledge_bases[selected_category], 
                    original_query_text, query_text_for_llm, 
                    metadata_filter, app_backend.HEADLINE_SUMMARIZER_PROMPT, # Note: HEADLINE_SUMMARIZER_PROMPT is used for synthesis here
                    app_backend.SYSTEM_INSTRUCTION_PROMPT, selected_category
                )
                # Stream the LLM response chunks
                for chunk_content in llm_response_generator:
                    response_queue.put({"type": "llm_chunk", "content": chunk_content})

                query_results["citations"] = convert_docs_to_dicts(retrieved_docs_for_ui) # Convert and pass citations
                
            elif selected_category in ['misc', 'sec', 'legal', 'news_full']:
                # handle_general_query returns LLM response and retrieved docs
                llm_response_generator, retrieved_docs_for_ui = app_backend.handle_general_query(
                    llm_instance, knowledge_bases, original_query_text, 
                    query_text_for_llm, selected_category, app_backend.SYSTEM_INSTRUCTION_PROMPT
                )
                # Stream the LLM response chunks
                for chunk_content in llm_response_generator:
                    response_queue.put({"type": "llm_chunk", "content": chunk_content})

                query_results["citations"] = convert_docs_to_dicts(retrieved_docs_for_ui) # Convert and pass citations
                
            elif selected_category == 'auto': # Fallback for general queries not caught by summary detection
                content_kbs_for_auto_query = [
                    kb for kb in knowledge_bases if kb not in [app_backend.NEWS_SUMMARIES_KB_NAME, app_backend.BLOG_SUMMARIES_KB_NAME, app_backend.SOCIAL_MEDIA_SUMMARIES_KB_NAME]
                ]
                llm_response_generator, retrieved_docs_for_ui = app_backend.handle_general_query(
                    llm_instance, knowledge_bases, original_query_text, 
                    query_text_for_llm, content_kbs_for_auto_query, 
                    app_backend.SYSTEM_INSTRUCTION_PROMPT
                )
                # Stream the LLM response chunks
                for chunk_content in llm_response_generator:
                    response_queue.put({"type": "llm_chunk", "content": chunk_content})

                query_results["citations"] = convert_docs_to_dicts(retrieved_docs_for_ui) # Convert and pass citations
            
            else:
                query_results["llm_response"] = f"Could not determine query type or find relevant knowledge base for '{input_query}'. Please try 'help' for options."
                response_queue.put({"type": "llm_chunk", "content": query_results["llm_response"]}) # Send message

        except Exception as e:
            query_results["llm_response"] = f"An error occurred: {e}"
            print(f"Error processing query: {e}")
            response_queue.put({"type": "llm_chunk", "content": query_results["llm_response"]}) # Send error to UI
        finally:
            response_queue.put({"type": "end_stream", "results": query_results})

    threading.Thread(target=process_query_thread, args=(user_input,)).start()
    return jsonify({"status": "processing"}) # Acknowledge the request immediately

@app.route('/stream_response')
def stream_response():
    """Streams LLM and reference responses to the frontend."""
    def generate():
        while True:
            item = response_queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] == "end_stream":
                break
            time.sleep(0.1) 
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("[web_ui.py] Starting web_ui.py. RAG system initialization running in background.")
    # Ensure data directories exist before app.py tries to load them
    app_backend.BASE_DATA_DIR.mkdir(exist_ok=True)
    app_backend.NEWS_DIR.mkdir(exist_ok=True)
    app_backend.MISC_DIR.mkdir(exist_ok=True)
    app_backend.SEC_DIR.mkdir(exist_ok=True)
    app_backend.LEGAL_DIR.mkdir(exist_ok=True)
    app_backend.BLOG_DIR.mkdir(exist_ok=True)
    app_backend.SOCIAL_MEDIA_DIR.mkdir(exist_ok=True)
    app_backend.BASE_VECTOR_DB_DIR.mkdir(exist_ok=True)

    try:
        app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False) # use_reloader=False to prevent double initialization
    finally:
        # Clean up llamafile process on shutdown
        if llamafile_process and isinstance(llamafile_process, subprocess.Popen):
            print("[web_ui.py] Terminating llamafile server from web_ui.py shutdown...")
            llamafile_process.terminate()
            llamafile_process.wait(timeout=5)
            if llamafile_process.poll() is None:
                print("[web_ui.py] Llamafile process did not terminate gracefully, forcing kill.")
                llamafile_process.kill()

