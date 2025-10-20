from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

# embedding manager
class EmeddingManager:
    def __init__(self, model_name:str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try: 
            self.model = SentenceTransformer(self.model_name)
            print(f"dimensions of model is {self.model.get_sentence_embedding_dimension()}")
        except Exception as e: 
            print(f"There was some error while loading the model {e}")

    def generate_embeddings(self, texts: List[str])-> np.ndarray:
        if not self.model:
            raise ValueError("Model not yet loaded")
        
        embeddings = self.model.encode(texts,show_progress_bar= False)
        return embeddings