from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.output_parser import StrOutputParser
import streamlit as st
import tempfile
import os

st.title('reportAnalyzer')

prompt_template = """Use the context below to write a 400 word summary about the topic below:
    Context: {context}
    Topic: {topic}
    Blog post:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "topic"])

llm = CTransformers(
                model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
                model_type="llama",
                config = {'context_length' : 2048}
                )

outputparser = StrOutputParser()
chain = LLMChain(llm=llm, prompt=PROMPT, output_parser=outputparser)
                 
uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

# loader=PyPDFLoader("./Kearney2019.pdf")
# doc = loader.load()

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
# print(doc)
# print(len(doc))
if uploaded_file is not None:
    doc = pdf_to_document(uploaded_file)
    document_splitter=CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=100)
    document_chunks=document_splitter.split_documents(doc)
    # print(document_chunks[0])
    # print(len(document_chunks))

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')
    vectordb.persist()

    def generate_blog_post(topic):
        # retriever = vectordb.as_retriever()
        # docs = retriever.invoke(
        #     topic
        # )
        # result = docs[0].page_content
        # print(result)
        docs = vectordb.similarity_search(topic, k=1)
        inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
        print(chain.apply(inputs))
        result = chain.apply(inputs)[0]["text"]
        return result

    st.header("Summary for the following topic")
    topic = st.text_input('topic?')

    if st.button('summary'):
        with st.spinner('Wait for it...'):
            st.write(generate_blog_post(topic))
