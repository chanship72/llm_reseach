import os
import chromadb

from more_itertools import batched
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


n_gpu_layers = 8  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

pdf_folder_path = "./TSLA/"
documents = []
for file in os.listdir(pdf_folder_path):
    # print(file)
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load_and_split())

# print(documents[205])

#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    # is_separator_regex = False,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunked_documents = text_splitter.split_documents(documents)
document_indices = list(range(len(chunked_documents)))

print("chunked_documents:",len(chunked_documents))

# model_name = 'NousResearch/Llama-2-7b-hf'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

client = chromadb.PersistentClient(path="./TSLA_vector")

if not client.list_collections():
    tsla_collection = client.create_collection(
        name="TSLA",
        embedding_function=embeddings
        )
    
    for i in range(len(chunked_documents)):
        tsla_collection.add(
            ids=str(i),
            documents=chunked_documents[i].page_content,
            metadatas=chunked_documents[i].metadata
        )
else:
    print("Collection already exists")
    strtsla_collection = client.get_collection(name="TSLA")

# print(tsla_collection.peek())
# for batch in batched(document_indices, 166):
#     start_idx = batch[0]
#     end_idx = batch[-1]
#     print(chunked_documents[start_idx:end_idx])
#     idsList = [str(i) for i in list(range(start_idx,end_idx))]
#     tsla_collection.add(
#         ids=idsList,
#         documents=chunked_documents[start_idx:end_idx]
#     )

vectordb = Chroma(persist_directory="./TSLA_vector", collection_name="TSLA", embedding_function=embeddings)
# docs = vectordb.similarity_search("core technology")
# print(docs)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# for i in range(len(retrieved_docs)):
#     print(retrieved_docs[i].page_content)
# print(chunked_documents[1].page_content)
# print(chunked_documents[1].metadata)

# vectordb = Chroma.from_documents(
#     documents=chunked_documents,
#     embedding=embeddings,
#     persist_directory='./TSLA_vector'
# )

# vectordb.persist()

# client = chromadb.PersistentClient(path="./TSLA_vector")
# tsla_collection = client.get_collection("TSLA")
# # tsla_collection = client.get_collection(name="TSLA")
# print(tsla_collection.peek())

# print(client.list_collections())

# Make sure the model path is correct for your system!
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
    model_type="llama",
    config = {'context_length' : 2048, 'gpu_layers' : 200}
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# response = chain.invoke("what's the core technology in Tesla?")
response = chain.invoke("what's the trend in financial summary in Tesla?")

print(response)
# # Prompt
# prompt = PromptTemplate.from_template(
#     "Summarize the main themes in these retrieved docs: {docs}"
# )

# # Chain
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # Build vectorstore and keep the metadata
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# vectorstore = Chroma(persist_directory="./data", embedding_function=embeddings)

# # Run
# question = "ESG"
# docs = vectorstore.similarity_search(question)
# result = llm_chain(docs)

# # Output
# print(result["text"])

