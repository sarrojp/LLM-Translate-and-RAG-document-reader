from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (using OpenAI)
client = OpenAI(
    api_key=openai_api_key,
    model="gpt-4.1-mini"
)


# Initialize FastAPI client
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)


# Create class with pydantic BaseModel
class TranslationRequest(BaseModel):
    input_str: str


# Function to set up the Translation system
def translate_text(input_str):
    prompt = f"""
    You are an expert translator who translates text from English to French
    and only return translated text.

    Text:
    {input_str}
    """

    return client.invoke(prompt)


# Function to set up the RAG system
def setup_rag_system():
    
    # Load the document
    loader = TextLoader('app/data/my_document.txt')
    documents = loader.load()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = splitter.split_documents(documents)

    # Initialize embeddings with OpenAI API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create FAISS vector store from document chunks and embeddings
    vector_store = FAISS.from_documents(document_chunks, embeddings)

    # Return the retriever for document retrieval with specified search_type
    retriever = vector_store.as_retriever(
        search_type="similarity",  # or "mmr" or "similarity_score_threshold"
        search_kwargs={"k": 5}  # Adjust the number of results if needed
    )
    return retriever


# Function to get the response from the RAG system
async def get_rag_response(query: str):

    retriever = setup_rag_system()
    
    # Retrieve the relevant documents using 'get_relevant_documents' method
    retrieved_docs = retriever.get_relevant_documents(query)

    # Prepare the input for the LLM: Combine the query and the retrieved documents into a single string
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # LLM expects a list of strings (prompts), so we create one by combining the query with the retrieved context
    prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]

    # Generate the final response using the language model (LLM)
    generated_response = client.invoke(prompt[0])
    
    return generated_response


# SET UP ROUTING   

# Route for translation
@app.post("/translate", tags=["Translation"])
async def translate(request: TranslationRequest):
    try:
        # Call your translation function
        translated_text = translate_text(request.input_str)
        return {"translated_text": translated_text}
    except Exception as e:
        # Handle exceptions or errors during translation
        raise HTTPException(status_code=500, detail=str(e))


# Route for RAG system query
@app.get("/query", tags=["RAG System"])
async def query_rag_system(query: str):
    try:
        # Pass the query string to your RAG system and return the response
        response = await get_rag_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))