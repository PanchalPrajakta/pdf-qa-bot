from fastapi import FastAPI, UploadFile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import uvicorn,tempfile
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "running"}

@app.post("/ask")
async def ask_question(file: UploadFile, query: str):
    # loader = PyPDFLoader(file.file.fileno())
    # docs = loader.load()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    print("docs",docs)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print("**splits",splits)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(splits, embedding_model)
    retriever = db.as_retriever()

    llm = Ollama(model="llama3.1:8b", base_url="http://localhost:11435")

    retrieval_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = retrieval_qa.run(query)

    return {"answer": result}


if __name__ == "__main__":
    uvicorn.run("rag:app", host="0.0.0.0", port=8070)