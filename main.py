
import torch

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from ctransformers import AutoModelForCausalLM,AutoConfig
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
load_dotenv()

st.title('arxivAnalyzer')

#파일 업로드
uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

st.write("---")
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 20,
        length_function = len,
        # is_separator_regex = False,
    )
    # texts = text_splitter.split_documents(pages)
    # print(texts)

    # split it into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(pages)
    print(len(docs))
    #Embedding
    # embeddings_model = OpenAIEmbeddings()
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # load it into Chroma
    db = Chroma.from_documents(docs, embeddings_model)

    # retriever = db.as_retriever(
    #     search_type="mmr",  # Also test "similarity"
    #     search_kwargs={"k": 8},
    # )

    # db = Chroma.from_documents(docs, OpenAIEmbeddings(disallowed_special=()))
    # for split_docs_chunk in texts:
    #     db = Chroma.from_documents(
    #             documents=split_docs_chunk, 
    #             embedding=embeddings_model,
    #             persist_directory='./vectordb'
    #         )
    #     db.persist()

    #Question
    st.header("tell me what you want!!")
    question = st.text_input('Question?')

    # config = AutoConfig.from_pretrained("llama-2-7b-chat.ggmlv3.q8_0.bin")
    # # Explicitly set the max_seq_len
    # config.max_seq_len = 4096
    # config.max_answer_len= 1024
    if st.button('Ask'):
        with st.spinner('Wait for it...'):
            llm = CTransformers(
                model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
                model_type="llama"
                )
            
            # llm = AutoModelForCausalLM.from_pretrained(
            #     "TheBloke/Llama-2-7B-GGUF",
            #     hf=True, 
            #     model_type="llama",
            #     gpu_layers=50,
            #     config=config
            # )
            # qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever(), chain_type="stuff")
            # st.write(qa_chain.run(question))

            qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
            qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=db.as_retriever())
            result = qa.run({"query": question})
            # st.write(result["result"])
            st.write(result)

            # retriver_from_llm = MultiQueryRetriever.from_llm(
            #     retriever=db.as_retriever(), llm=llm
            # )

            # docs = retriver_from_llm.get_relevant_documents(query=question)
            # print(len(docs))
            # print(docs)

            # memory = ConversationSummaryMemory(
            #     llm=llm, memory_key="chat_history", return_messages=True
            # )
            # qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
            # result = qa(question)
            # st.write(result["answer"])
# template = """Question: {question}
# Answer:"""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm = CTransformers(
#     model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
#     model_type="llama"
#     )
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# response = llm_chain.run("coding")

# st.write(response)
# llm = OpenAI()
# chat_model = ChatOpenAI()

# llm.predict("hi!")

# loader = PyPDFLoader("2005.11401.pdf")
# pages = loader.load_and_split()


# print(pages[0].page_content)