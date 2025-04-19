from vector_db import VectorDB
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class QueryProcessor:
    def __init__(self, vector_db: VectorDB):
        """Initialize query processor with vector DB."""
        self.vector_db = vector_db
        self.llm = ChatOllama(model="qwen:0.5b", temperature=0.)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = None
        self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        """Set up the conversational retrieval chain."""
        retriever = self.vector_db.get_retriever(k=3)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )

    def ask_question(self, question: str) -> str:
        """Query the RAG system with a question."""
        try:
            result = self.qa_chain({"question": question})
            return result["answer"]
        except Exception as e:
            print(f"Error processing question: {e}")
            return "An error occurred while processing the question."
        
    def get_top_k_results(self, query: str, k: int = 3) -> list:
        """Retrieve the top k document chunks for a query using the retriever."""
        try:
            retriever = self.vector_db.get_retriever(k=k)
            results = retriever.invoke(query)
            return [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            print(f"Error retrieving top k results: {e}")
            return []