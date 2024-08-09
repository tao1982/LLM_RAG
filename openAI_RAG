import openai
import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

from sentence_transformers import SentenceTransformer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

import torch
import os




torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

persist_directory ='/home/jpei/chromaTraining'

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(
    openai_api_key='',
    model="text-embedding-3-large"
)

#embeddings = OllamaEmbeddings( model="llama3.1")
vectordb  = Chroma(persist_directory = persist_directory,embedding_function = embeddings)
def format_docs(docs):
    infos = "\n\n"
    for doc in docs:
        #infos = infos + doc.page_content + " Bezug:"+ doc.metadata['source'] + "\n\n"
        infos = infos + doc.page_content + "<br>###################################################<br>"
    #return "\n\n".join((doc.page_content + doc.metadata) for doc in docs)
    print(infos)
    return infos
##Define the Ollama LLm function
def openAI_Chat(question,context):

    openai.api_key = ""
    #response = openai.chat.Completion.create(
    # engine="gpt-3.5-turbo-16k",
    # prompt="suchen das Antwort für die Frage:'{question}' in diesem Kontext:'{context}' auf Deutsch \n\n",
    # max_tokens=50
    # )

    #print(response.choices[0].text.strip())

    completion = openai.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages=[
   # {"role": "system", "content":"find the Answser for:'{question}' in this context:  '{context}'  \n\n" },
   {"role": "user", "content":f"find the Answer for:{question} in this context:{context} auf deutsch.\n\n"}
      ]
     )

    return "chromaQuery:" + context + "<br><br><br> OpenAI:"+ completion.choices[0].message.content
   # return response.choices[0].text.strip()
##Define the RAG chain
def ollama_Chat(question, context):
    formatted_prompt = f"find the Answser for :{question} in this context:{context}.\n\n"
    #response = ollama.chat(model ='sroecker/sauerkrautlm-7b-hero:latest', messages=[{'role':'user','content':formatted_prompt}])
    response = ollama.chat(model ='llama3.1:latest', messages=[{'role':'user','content':formatted_prompt}])

    return "chromaQuery:" + context + "<br><br><br> OllamaAI:" + response['message']['content']

def rag_chain(question,aiModel):
   # retrieved_docs = retriever.invoke(question)
    print("question:")
    print(question)
    retrieved_docs = vectordb.similarity_search(f"{question}",5)
    formatted_context = format_docs(retrieved_docs)
   # formatted_context = retrieved_docs
    print("retrieved_docs:")
    print(retrieved_docs)
    if aiModel== "openai":
     return openAI_Chat(question, formatted_context)
    else:
     return ollama_Chat(question, formatted_context)
#Use the RAG chain
app = Flask(__name__)
@app.route('/neuTrainingsDatei', methods=['GET','POST'])
def addNeuTrainingsDatei():
    if 'file' not in request.files:
        return ("error: No file part 400 ")

    file = request.files['file']

    if file.filename == '':
        return ("error: No selected file 400 ")

    if file:
        # Save the file temporarily
        filepath = os.path.join("/tmp",file.filename)
        print(filepath)
        filename, file_extension = os.path.splitext(filepath)
        print(filename)
        print(file_extension)
        if file_extension!=".txt" and file_extension!=".pdf":
            return ("Error: die Dateitype ist nicht erlaut.")
else:
            file.save(filepath)

        # Get the size of the file
        file_size = os.path.getsize(filepath)
    if file_extension==".txt":
       loader = TextLoader(filepath)
       print(filepath)

    if file_extension==".pdf":
       loader = PyPDFLoader(filepath)
       print(filepath)

    documents = loader.load()
    print(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        separators=[
        "\n\n"
        ],
    )
    texts = text_splitter.split_documents(documents)
    print(texts)
    persist_directory = '/home/jpei/chromaDBpersist'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})
    #embeddings = OllamaEmbeddings(model="llama3:70b")

    vectordb  = Chroma(persist_directory = persist_directory,embedding_function = embeddings)
   # vectordb  = Chroma.from_documents(documents = texts, embedding = embeddings,persist_directory = persist_directory)

    vectordb.add_documents(texts)

    fileInfos = filepath
    os.remove(filepath)
    return jsonify({fileInfos: "wurde hochgeladen."})
@app.route('/imotionchatbotOpenAI', methods=['GET','POST'])
def kiAntwortOpenAI():
    qa =request.args.get('chat')
    print("qa")
    print(qa)
    antwort = rag_chain(qa,"openai")
    return antwort

@app.route('/imotionchatbot', methods=['GET','POST'])
def kiAntwortOllama():
    qa =request.args.get('frage')
    print("qa")
    print(qa)
    antwort = rag_chain(qa,"ollama")
    return antwort

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8443)
    http_server = WSGIServer(("127.0.0.1", 8443), app)
    http_server.serve_forever()

#result = rag_chain("Abholung bei I-Motion beauftragen?")
#print("Result:")
#print(result)
