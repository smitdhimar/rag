from loader import get_chunks
from embedder import EmeddingManager
from retriever import RAGretriever
from vector_store import VectorStore

data_path = "../data/pdf"
presistent_directory = "../vector_store"

chunks = get_chunks(data_path, 1000, 200)

embedding_manager = EmeddingManager()
print(chunks[0])
generated_embeddings = embedding_manager.generate_embeddings([i.page_content for i in chunks])
print(generated_embeddings)

vector_store= VectorStore()
vector_store.add_documents(chunks, generated_embeddings)

retirever = RAGretriever(vector_store, embedding_manager)
response = retirever.retrievel("what is professional reference letter")
print(response)