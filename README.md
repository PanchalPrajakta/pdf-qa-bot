FastAPI backend and a simple HTML/JS for a PDF Question-Answering system. Upload a PDF and ask questions; answers are generated using embeddings, FAISS vector search, and Ollama (Llama 3.1:8B).

# Features

- Upload any PDF and preview it in the browser.
- Ask questions about the content.
- Extract and chunk text using LangChain's text splitter.
- Generate embeddings with HuggingFace's all-MiniLM-L6-v2 model.
- Powered by Ollama running locally.

# Project Structure
RAG/
- │── rag.py              # FastAPI backend (RAG pipeline)
- │── requirements.txt    # Python dependencies
- │── frontend.html       # UI (upload, ask, answer)
- │── README.md           # Project documentation

# Installation
1. Clone Repo
    git clone https://github.com/your-username/pdf-qa-bot.git
    - cd pdf-qa-bot/RAG

2. Create Virtual Environment
   - python -m venv venv
   - source venv/bin/activate  (Linux/Mac)
   - venv\Scripts\activate
   
4. Install Dependencies
    pip install -r requirements.txt

# Running the APP
1. Start Ollama
   Make sure Ollama installed and the Llama model is available.
   
   - ollama pull llama3.1:8b
   - ollama run llama3.1:8b

2. From the RAG/ folder
   - uvicorn rag:app --host 0.0.0.0 --port 8070 --reload

   API will be available at:
   - http://localhost:8070

3. Open the Frontend
   - Navigate to frontend.html and open in a browser.

# Example workflow
1. Upload contract.pdf via frontend.
2. Ask question.
3. Backend:
     - Extracts and chunks text
     - Embeds with HuggingFace
     - Stores/searches in FAISS
     - Sends context to Ollama (Llama 3.1:8b)
4. Answer is displayed in the frontend.
   
