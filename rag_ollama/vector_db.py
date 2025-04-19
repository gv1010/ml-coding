import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, vectorstore_dir: str = "./faiss_index"):
        """Initialize FAISS vector database."""
        self.vectorstore_dir = vectorstore_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-large-Embedding-v0")
        self.vectorstore = None

    def load_or_create_vectorstore(self, texts: list):
        """Load existing vector store or create a new one from texts."""
        try:
            if os.path.exists(self.vectorstore_dir):
                print(f"Loading existing vector store from {self.vectorstore_dir}...")
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                print(f"Creating new vector store and saving to {self.vectorstore_dir}...")
                self.vectorstore = FAISS.from_documents(texts, self.embeddings)
                self.save_vectorstore()
        except Exception as e:
            print(f"Error with vector store: {e}")
            raise SystemExit("Failed to create or load vector store. Ensure the GIST-large-Embedding-v0 model is accessible.")

    def save_vectorstore(self):
        """Save the vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(self.vectorstore_dir)

    def get_retriever(self, k: int = 3):
        """Return a retriever for the vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Ensure documents are indexed by calling load_and_index_documents in IndexManager.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

# import os
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings

# class VectorDB:
#     def __init__(self, vectorstore_dir: str = "./faiss_index_ramayana"):
#         """Initialize FAISS vector database."""
#         self.vectorstore_dir = vectorstore_dir
#         self.embeddings = OllamaEmbeddings(model="qwen:0.5b")
#         self.vectorstore = None

#     def load_or_create_vectorstore(self, texts: list):
#         """Load existing vector store or create a new one from texts."""
#         try:
#             if os.path.exists(self.vectorstore_dir):
#                 print(f"Loading existing vector store from {self.vectorstore_dir}...")
#                 self.vectorstore = FAISS.load_local(
#                     self.vectorstore_dir,
#                     self.embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#             else:
#                 print(f"Creating new vector store and saving to {self.vectorstore_dir}...")
#                 self.vectorstore = FAISS.from_documents(texts, self.embeddings)
#                 self.save_vectorstore()
#         except Exception as e:
#             print(f"Error with vector store: {e}")
#             raise SystemExit("Failed to create or load vector store. Ensure Ollama is running and the Qwen:0.5b model is available.")

#     def save_vectorstore(self):
#         """Save the vector store to disk."""
#         if self.vectorstore:
#             self.vectorstore.save_local(self.vectorstore_dir)

#     def get_retriever(self, k: int = 3):
#         """Return a retriever for the vector store."""
#         if not self.vectorstore:
#             raise ValueError("Vector store not initialized.")
#         return self.vectorstore.as_retriever(search_kwargs={"k": k})