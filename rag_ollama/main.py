from vector_db import VectorDB
from index_manager import IndexManager
from query_processor import QueryProcessor
import logging
logging.captureWarnings(True)

# Initialize components
vector_db = VectorDB(vectorstore_dir="./faiss_index")
index_manager = IndexManager(docs_dir="./documents")
index_manager.load_and_index_documents(vector_db)
query_processor = QueryProcessor(vector_db=vector_db)

# Load and index documents
try:
    index_manager.load_and_index_documents(vector_db)
except SystemExit as e:
    print(e)
    exit(1)

# Ask questions
questions = [
    "What is this article about?",
    "What did he say about job disruption?",
]
for question in questions:
    print(f"\nQuestion: {question}")
    top_k_results = query_processor.get_top_k_results(question, k=3)
    print("Top k results:")
    for i, result in enumerate(top_k_results, 1):
        print(f"  {i}. Text: {result['text'][:100]}... (Metadata: {result['metadata']})")
    answer = query_processor.ask_question(question)
    print("\n "*3)
    print(f"Answer: {answer}")