#from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import VectorDBQA


from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
#import chromadb
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from sentence_transformers import SentenceTransformer

from langchain_huggingface import HuggingFaceEmbeddings
import torch

#from flask import Flask, request, jsonify
#import json
#import os
#from gevent.pywsgi import WSGIServer

#from langchain.text_splitter import CharacterTextSplitte

# Load and process the text

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


loader = TextLoader("/home/jpei/imoKiTraining.txt")
documents = loader.load()
#print(documents)
text_splitter = RecursiveCharacterTextSplitter(
 chunk_size=1000,
 chunk_overlap=200,
        separators=[
        "\n\n",
        "\n"
        "."
        "!"
        "?"
       ],
    )
    #text_splitter = RecursiveCharacterTextSplitter()
texts = text_splitter.split_documents(documents)
#print(texts)
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk

persist_directory = '/home/jpei/chromaTraining'

    #embeddings  = OllamaEmbeddings(model='llama3.1')
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = (OllamaEmbeddings())
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(
     openai_api_key='',
     model="text-embedding-3-large"
    )
    #embeddings = OllamaEmbeddings(model="llama3:70b")
    #embeddings = OllamaEmbeddings(model='mxbai-embed-large:latest')
vectordb  = Chroma.from_documents(documents = texts, embedding = embeddings,persist_directory = persist_directory  )
#vectordb  = Chroma(persist_directory = persist_directory,embedding_function = embeddings)
#vectordb.add_documents(texts)

results = vectordb.similarity_search("wo kann ich die Lieferankündigungen sehen?",3)

    #retriever = vectordb.as_retriever()
    #results   = retriever.invoke('Meine SMC-B läuft ab. Bestellt I-Motion automatisch eine neue für mich?')
print(results)
