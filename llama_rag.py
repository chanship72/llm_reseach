import streamlit as st
import tempfile
import os

# from dotenv import load_dotenv
# load_dotenv(verbose=True)
# openai_key = os.environ.get('OPENAI_API_KEY')

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings

from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

n_gpu_layers = 8  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


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
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma(persist_directory="./chroma").from_documents(texts, embeddings_model, collection_name="lecture_stt")

    #Question
    st.header("강사님께 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            chroma_retriever = db.as_retriever()

            retrieved_docs = chroma_retriever.invoke(
                question
            )
            page_content = retrieved_docs[0].page_content.replace("\t","").replace("\n","")
            metadata = str(retrieved_docs[0].metadata)
            st.write(f"page_content: :blue[{page_content}]")
            st.write(f"metadata: :blue[{metadata}]")

            llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", temperature=0)
            # llm = ChatOpenAI(model_name="gpt-4", temperature=0)

            qa_chain = RetrievalQA.from_chain_type(llm,retriever=chroma_retriever, verbose=True)
            rag_result = qa_chain({"query": question})
            gpt_result = llm.predict(question)
            tab1, tab2 = st.tabs(['RAG', 'LLAMA2-7B'])
            with tab1:
                st.header('RAG')
                st.write(rag_result["result"])
            with tab2:
                st.header('LLAMA2-7B')
                st.write(gpt_result)