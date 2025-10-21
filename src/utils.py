# prepare llm class for sending messages
import requests
from retriever import RAGretriever
class OllamaLLM:
    def __init__(self, model_name:str="gemma3:1b", host_name:str="http://localhost:11434/api/generate", is_stream:bool=False):
        self.model_name = model_name
        self.host_name = host_name
        self.is_stream = is_stream

    def send_message(self, rag_retriever: RAGretriever, question: str):
        try:
            contexts = rag_retriever.retrievel(question, 3, 0.3);
            payload = {
                "model": self.model_name,
                "prompt": f"""Given the context and the user question return the answer of the user question by referring context, if enough details are not found in context try to elaborate in your own way \n context: { "\n".join(i["content"] for i in contexts) }
 \n
                question: {question} \n""" ,
                "stream": self.is_stream
            }
            response = requests.post(self.host_name, json=payload)
            if(response.status_code == 200):
                data = response.json();
                return data
            else:
                print(f"There is some error in api : {response}")
                return {}
        except Exception as e:
            print(f"There is some error : {e}")