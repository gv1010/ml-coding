# RAG with Ollama and FIASS
##### Installation

Install Ollama:
```
curl -fsSL https://ollama.com/install.sh | sh
```


Pull the Qwen:0.5b Model:  
https://ollama.com/library/qwen:0.5b


```
ollama pull qwen:0.5b

```

Embedding Model:  
https://huggingface.co/thenlper/gte-base

```
self.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

sequence length = 512
embed_size = 768
```



```
vector_db.py
    - this loads the text embedding model, indexes the documents, or loads exsiting indexed file.

index_manager.py
    - this prepares the text files for indexing by chunking the text documents.
    - calls the VectorDB class to index the documents.
query_processor.py
    - retriever is loaded from vector_db to return top_k=3 relevant results.
    - conversational retrieval chain is used to combine, retreivier output with LLMs (qwen) to augement the response.
    - conversational buffer is used to maintain context-aware responses.

main.py
    -  code from above files are managed in main.py
    -  this prints, the top_k = 3 reuslts for the given query.
    -  also the final llm response.
```


##### Running the Application

Run the main.py Script:
- If faiss_index folder is not found, it indexes .txt files in the documents folder.
- If faiss_index exists, it loads the existing index.

```python3 main.py```




##### to be added if the systems needs to scaled:
1. host vector db as a service, which can handle indexing, distributed search.
2. for indexing new documents and update the vector db, this avoids reindexing the existing documents indexing.




