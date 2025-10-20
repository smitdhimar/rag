import chromadb
import uuid
import numpy as np
from typing import List, Any
import os

class VectorStore:
    def __init__(
            self,
            collection_name:str = 'demo-vector-store',
            persistent_directory: str = "../vector_store"
            ):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.persistent_directory = persistent_directory
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persistent_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistent_directory)

            # get or create collection 
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={
                    "description": "PDF document embeddings for rag",
                    "hnsw:space":"cosine"
                }
            )
            print("Initialized")
            print(f"Existing documents in collection are: {self.collection.count()}")
        except Exception as e:
            print(f"There was some error while initiating vector store: {e}")

    def add_documents(self, documents:List[any], embeddings: np.ndarray):
        ids = []
        embeddings_list = []
        document_texts = []
        metadatas = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i 
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            document_texts.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

            try:
                self.collection.add(
                    ids = ids,
                    embeddings= embeddings_list,
                    metadatas= metadatas,
                    documents= document_texts
                )
                print(f"documents added");

            except Exception as e:
                print(f"Error adding documents: {e}")