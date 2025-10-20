from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
def load_directory(path: str):
    directory_loader= DirectoryLoader(
        path= path,
        glob= "**/*.pdf",
        loader_cls= PyPDFLoader
    )
    directory_documents = directory_loader.load()
    return directory_documents

def split_documents(directory_documents: List[Any], chunk_size:int=1000, chunk_overlap: int=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap = chunk_overlap,
        chunk_size = chunk_size,
        length_function = len,
        separators= ["\n\n", "\n"]
    )
    split_docs = text_splitter.split_documents(directory_documents)
    print(f"split {len(directory_documents)} into {len(split_docs)} chunks")
    return split_docs

def get_chunks(path: str, chunk_size: int = 1000, chunk_overlap: int= 200):
    # 1. load directory and get documents from it
    documents = load_directory(path)

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    return chunks