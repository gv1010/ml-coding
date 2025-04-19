import os
import nltk
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_db import VectorDB

class IndexManager:
    def __init__(self, docs_dir: str = "./documents", chunk_size: int = 500, chunk_overlap: int = 100):
        """Initialize index manager with document directory and splitter."""
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_index_documents(self, vector_db: VectorDB):
        """Load documents, split, and index into vector DB."""
        # Ensure documents directory exists
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            sample_content = """Sample document for RAG.
                                This is a test document to demonstrate the RAG system.
                                It contains some basic information about LangChain and Ollama."""
            with open(os.path.join(self.docs_dir, "sample.txt"), "w") as f:
                f.write(sample_content)
            print(f"Created sample document in {self.docs_dir}/sample.txt")

        # Load documents
        try:
            loader = DirectoryLoader(self.docs_dir, glob="*.txt", loader_cls=TextLoader)
            documents = loader.load()
            if not documents:
                raise ValueError("No documents found in the specified directory.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise SystemExit("Failed to load documents. Please ensure the documents directory contains valid .txt files.")

        # Split documents into chunks
        try:
            texts = self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error splitting documents: {e}")
            raise SystemExit("Failed to split documents. This may be due to NLTK issues or invalid document content.")

        # Index documents into vector DB
        vector_db.load_or_create_vectorstore(texts)