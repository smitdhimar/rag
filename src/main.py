#imports
from utils import OllamaLLM
from loader import get_chunks
from embedder import EmeddingManager
from retriever import RAGretriever
from vector_store import VectorStore

#declare paths
data_path = "../data/pdf"
presistent_directory = "../vector_store"

# get chunks 
chunks = get_chunks(data_path, 1000, 200)

# initialize embedding manager
embedding_manager = EmeddingManager()
generated_embeddings = embedding_manager.generate_embeddings([i.page_content for i in chunks])

# initialize vector stores
vector_store= VectorStore()
is_add_document_enabled = False
if is_add_document_enabled is True:
    vector_store.add_documents(chunks, generated_embeddings)

# retriever class
retirever = RAGretriever(vector_store, embedding_manager)
# retreive context
# response = retirever.retrievel("what is professional reference letter")

# ollama llm connection
question = "what is professional reference letter"
llm = OllamaLLM()
response = llm.send_message(retirever, question)
print(response['response'])