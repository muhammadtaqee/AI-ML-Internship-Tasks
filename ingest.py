"""
Document Ingestion Module for RAG Chatbot
Handles loading, processing, and vectorizing documents for retrieval
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentIngestionPipeline:
    """
    Handles document loading, chunking, and vector store creation
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        vector_store_path: str = "vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the document ingestion pipeline
        
        Args:
            data_dir: Directory containing source documents
            vector_store_path: Path to save vector database
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model to use
        """
        self.data_dir = Path(data_dir)
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from data directory or specific files
        
        Args:
            file_paths: Optional list of specific file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        if file_paths:
            # Load specific files
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                docs = self._load_single_document(file_path)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} chunks from {file_path}")
        else:
            # Load all documents from data directory
            loaders = [
                DirectoryLoader(
                    str(self.data_dir),
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True
                ),
                DirectoryLoader(
                    str(self.data_dir),
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    show_progress=True,
                    loader_kwargs={"encoding": "utf-8"}
                ),
                DirectoryLoader(
                    str(self.data_dir),
                    glob="**/*.md",
                    loader_cls=UnstructuredMarkdownLoader,
                    show_progress=True
                )
            ]
            
            for loader in loaders:
                try:
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents")
                except Exception as e:
                    logger.error(f"Error loading with loader: {e}")
                    
        return documents
    
    def _load_single_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document based on its extension
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of documents from the file
        """
        extension = file_path.suffix.lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif extension == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return []
            
        return loader.load()
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to process")
            return []
            
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            
        return chunks
    
    def create_vector_store(self, chunks: List[Document], use_faiss: bool = False) -> object:
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of document chunks
            use_faiss: Whether to use FAISS (faster) or Chroma (persistent)
            
        Returns:
            Vector store object
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
            
        if use_faiss:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            # Save FAISS index
            vector_store.save_local(str(self.vector_store_path / "faiss_index"))
            logger.info(f"FAISS vector store created with {len(chunks)} chunks")
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.vector_store_path / "chroma_db")
            )
            vector_store.persist()
            logger.info(f"Chroma vector store created with {len(chunks)} chunks")
            
        return vector_store
    
    def load_existing_vector_store(self, use_faiss: bool = False) -> Optional[object]:
        """
        Load existing vector store if available
        
        Args:
            use_faiss: Whether to load FAISS or Chroma store
            
        Returns:
            Loaded vector store or None if not exists
        """
        if use_faiss:
            faiss_path = self.vector_store_path / "faiss_index"
            if faiss_path.exists():
                return FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
        else:
            chroma_path = self.vector_store_path / "chroma_db"
            if chroma_path.exists():
                return Chroma(
                    persist_directory=str(chroma_path),
                    embedding_function=self.embeddings
                )
        return None
    
    def run_pipeline(self, use_faiss: bool = False) -> object:
        """
        Run the complete ingestion pipeline
        
        Args:
            use_faiss: Whether to use FAISS for vector storage
            
        Returns:
            Vector store object
        """
        # Check if vector store already exists
        existing_store = self.load_existing_vector_store(use_faiss)
        if existing_store:
            logger.info("Using existing vector store")
            return existing_store
            
        # Load and process documents
        logger.info("Starting document ingestion pipeline...")
        documents = self.load_documents()
        
        if not documents:
            raise ValueError("No documents found in data directory")
            
        chunks = self.process_documents(documents)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
            
        # Create vector store
        vector_store = self.create_vector_store(chunks, use_faiss)
        
        logger.info("Document ingestion pipeline completed successfully")
        return vector_store

# Sample document creation for testing
def create_sample_documents():
    """
    Create sample documents for testing the chatbot
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_docs = [
        {
            "name": "ai_intro.txt",
            "content": """
            Artificial Intelligence (AI) is the simulation of human intelligence in machines 
            that are programmed to think and learn. AI can be categorized into:
            
            1. Narrow AI: Designed for specific tasks like facial recognition or language translation
            2. General AI: Systems with human-like intelligence across various domains
            3. Super AI: AI that surpasses human intelligence
            
            Key AI Technologies:
            - Machine Learning: Algorithms that learn from data
            - Deep Learning: Neural networks with multiple layers
            - Natural Language Processing: Understanding and generating human language
            - Computer Vision: Interpreting visual information
            
            Applications of AI:
            - Healthcare: Disease diagnosis, drug discovery
            - Finance: Fraud detection, algorithmic trading
            - Transportation: Autonomous vehicles, traffic prediction
            - Entertainment: Content recommendation, game AI
            """
        },
        {
            "name": "rag_explained.txt",
            "content": """
            Retrieval-Augmented Generation (RAG) is an AI framework that combines 
            retrieval systems with large language models to generate more accurate and 
            contextually relevant responses.
            
            How RAG Works:
            1. Query Processing: User question is processed
            2. Document Retrieval: Relevant documents are retrieved from vector database
            3. Context Augmentation: Retrieved documents are added to the prompt
            4. Generation: LLM generates response using the augmented context
            
            Benefits of RAG:
            - Reduced hallucinations through factual grounding
            - Access to up-to-date information without retraining
            - Transparency through source citation
            - Cost-effective compared to fine-tuning
            
            RAG vs Fine-tuning:
            RAG is better for knowledge-intensive tasks and when data changes frequently,
            while fine-tuning is better for style adaptation and specific formatting needs.
            """
        },
        {
            "name": "langchain_guide.txt",
            "content": """
            LangChain is a framework for developing applications powered by language models.
            It provides modular components for building LLM applications.
            
            Core Components:
            - Models: LLM wrappers (OpenAI, Anthropic, etc.)
            - Prompts: Template management and optimization
            - Chains: Combining multiple components
            - Agents: Decision-making systems
            - Memory: Conversation context management
            - Retrievers: Document retrieval systems
            
            Popular Use Cases:
            - Chatbots with memory
            - Document question answering
            - Code understanding and generation
            - Data analysis and visualization
            
            Integration with RAG:
            LangChain provides seamless integration with vector stores like Chroma,
            FAISS, and Pinecone, making it ideal for building RAG applications.
            """
        }
    ]
    
    for doc in sample_docs:
        file_path = data_dir / doc["name"]
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["content"].strip())
            logger.info(f"Created sample document: {file_path}")
        else:
            logger.info(f"Sample document already exists: {file_path}")

if __name__ == "__main__":
    # Create sample documents for testing
    create_sample_documents()
    
    # Initialize and run pipeline
    pipeline = DocumentIngestionPipeline(
        data_dir="data",
        vector_store_path="vector_db",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Run the pipeline
    vector_store = pipeline.run_pipeline(use_faiss=False)
    print("✅ Vector store created successfully!")
    print(f"📊 Vector store type: {type(vector_store).__name__}")
    