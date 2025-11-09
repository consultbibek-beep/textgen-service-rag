# mini-gen-search/textgen-service-rag/rag_api.py
from flask import request, jsonify
import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_client import QdrantClient, models

# RAG CONFIG
PDF_PATH = "/app/138.pdf"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "electoral_list_rag"
LLM_MODEL = "llama3-8b-8192"

# Global RAG chain (will be set by the main app)
rag_chain = None

def ingest_pdf(logger):
    """Load, chunk, embed, and index the PDF data into Qdrant idempotently."""
    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF file not found at {PDF_PATH}. RAG will not function.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="/app/embeddings_model") # Use local path for Docker
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Check if collection exists and is populated
        try:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            if collection_info.points_count > 0:
                logger.info(f"Collection '{COLLECTION_NAME}' populated. Skipping ingestion.")
                return Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
        except Exception:
            logger.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating and ingesting data.")
    except Exception as e:
        logger.error(f"Could not connect to Qdrant or initialize collection: {e}")
        return None

    # Ingestion logic
    logger.info("Starting PDF Ingestion and Indexing...")
    loader = UnstructuredFileLoader(PDF_PATH, strategy="hi_res", mode="elements")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    
    vector_store_instance = Qdrant.from_documents(
        chunks,
        embeddings,
        client=client,
        collection_name=COLLECTION_NAME
    )
    logger.info("✅ PDF Ingestion and Indexing complete.")
    return vector_store_instance

def setup_rag(vector_store_instance, logger):
    """Sets up the LangChain RAG retrieval chain using Groq."""
    global rag_chain
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key or not vector_store_instance:
        logger.error("RAG setup failed due to missing API key or vector store.")
        return None
        
    logger.info("Setting up RAG chain...")
    llm = ChatGroq(temperature=0, model_name=LLM_MODEL, api_key=groq_api_key)
    retriever = vector_store_instance.as_retriever(k=5)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert search assistant for an electoral roll document. "
         "Your sole purpose is to answer the user's question based ONLY on the "
         "following retrieved document chunks. If the answer is not in the "
         "documents, you MUST clearly state 'I could not find the information in the provided electoral list.' "
         "Retrieved Context: \n\n{context}"
        ),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    logger.info("✅ RAG chain setup complete.")
    return rag_chain

def setup_rag_route(app, rag_chain_instance):
    """Registers the /ask route to the Flask app."""
    
    @app.route('/ask', methods=['POST'])
    def ask_rag():
        """Handles the user's search query."""
        if not rag_chain_instance:
            return jsonify({"error": "RAG system not initialized. Check server logs."}), 500
            
        prompt = request.json.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400
            
        app.logger.info(f"Received RAG query: {prompt}")
        
        try:
            response = rag_chain_instance.invoke({"input": prompt})
            result = response.get("answer", "An unexpected error occurred during generation.")
            
            return jsonify({"generated": result})
        except Exception as e:
            app.logger.error(f"Error during RAG chain execution: {e}")
            return jsonify({"error": str(e)}), 500