# pdf-qa-bot
FastAPI backend and a simple HTML/JS for a PDF Question-Answering system. Upload a PDF and ask questions; answers are generated using embeddings, FAISS vector search, and Ollama (Llama 3.1:8B).

#Features
- Upload any PDF and preview it in the browser.
- Ask questions about the content.
- Extract and chunk text using LangChain's text splitter.
- Generate embeddings with HuggingFace's all-MiniLM-L6-v2 model.
- Powered by Ollama running locally.
  
