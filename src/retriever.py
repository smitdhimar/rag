from typing import List, Dict, Any
from vector_store import VectorStore
from embedder import EmeddingManager

class RAGretriever:
    def __init__(
            self,
            vector_store: VectorStore, embedding_manager: EmeddingManager
        ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrievel(self, prompt: str, top_k: int = 3, score_threshold:float = 0.3 ) -> List[Dict[str,Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([prompt])[0]

        try:
            results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()],
            n_results=top_k                             )
            retrieved_documents=[]

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i,(doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1-distance
                    if(similarity_score >= score_threshold):
                        retrieved_documents.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'distance': distance,
                            'rank': i+1
                        })

                print(f"Retrieved {len(retrieved_documents)} results")
            else:
                print("No documents retrieved")
            return retrieved_documents
        except Exception as e:
            print(f"Error: {e}")
            return []