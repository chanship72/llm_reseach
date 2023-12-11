import streamlit as st
import tempfile
import os

# from dotenv import load_dotenv
# load_dotenv(verbose=True)
# openai_key = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = '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'

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

#파일 업로드
uploaded_file = st.file_uploader("Choose a file")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def dat_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = TextLoader(temp_filepath,encoding="utf-8")
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    # pages = pdf_to_document(uploaded_file)

    pages = dat_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("강사님께 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            chroma_retriever = db.as_retriever()

            retrieved_docs = chroma_retriever.invoke(
                question
            )
            for i in range(len(retrieved_docs)):
                page_content = retrieved_docs[i].page_content.replace("\t","").replace("\n","")
                metadata = str(retrieved_docs[i].metadata)
                st.write(f"page_content: :blue[{page_content}]")
                st.write(f"metadata: :blue[{metadata}]")

            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=chroma_retriever, verbose=True)
            rag_result = qa_chain({"query": question})
            gpt_result = llm.predict(question)
            tab1, tab2 = st.tabs(['RAG', 'GPT4'])
            with tab1:
                st.header('RAG')
                st.write(rag_result["result"])
            with tab2:
                st.header('GPT4')
                st.write(gpt_result)