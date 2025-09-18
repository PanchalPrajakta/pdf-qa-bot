# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
# from langchain.agents import Tool, initialize_agent, AgentType
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import CharacterTextSplitter
# # from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader
# # 1. Documents
# # docs = [
# #     Document(page_content="The GDPR is a data privacy law enacted in the EU in 2018."),
# #     Document(page_content="India's Digital Personal Data Protection Act was passed in 2023."),
# #     Document(page_content="GDPR requires user consent and imposes heavy penalties."),
# #     Document(page_content="Indian law emphasizes data localization for companies."),
# # ]

# loader=PyPDFLoader('C:\Prajakta\ML\MongoDB.pdf')
# docs=loader.load()
# # 2. Split text into chunks
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# print("**text splitter",text_splitter)
# # print(len(text_splitter))

# splits = text_splitter.split_documents(docs)
# print("*** splits",splits)
# # 3. Embeddings + FAISS vectorstore
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# db = FAISS.from_documents(splits, embedding_model)
# print("***db",db)
# retriever = db.as_retriever()
# print("***retriever",retriever)
# # 4. LLM from Ollama
# llm = Ollama(model="llama3.1:8b",base_url="http://localhost:11435") 

# # 5. Retrieval QA chain
# retrieval_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# print("****retrieval",retrieval_qa)

# # 6. Wrap QA in a Tool
# retrieval_tool = Tool(
#     name="document_search",
#     func=retrieval_qa.run,
#     description="Use this tool to answer questions about PDF"
# )
# print("*****retrieval tool",retrieval_qa)

# system_prompt = """
# You are an intelligent agent that can take actions using tools. 
# Follow this format strictly:
# Thought: <your reasoning>
# Action: <tool_name>
# Action Input: <input to tool>
# """
# # 7. Agent with tool access
# agent = initialize_agent(
#     tools=[retrieval_tool],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=10, 
#     system_message=system_prompt
# )
# print("*****agent",agent)

# # 8. Ask question
# query = "What is in that document?"
# result = agent.run(query)
# print("****Answer:", result)





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