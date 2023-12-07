import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import torch
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain

from transformers import pipeline
# from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):    
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # Define prompt
    _template = """
    Given the following conversation and a follow up question, return answer in Korean.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Translates English to Korean.
    """
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    # llm = ChatOpenAI(temperature=0)
    llm = CTransformers(
                model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
                model_type="llama",
                config = {'context_length' : 2048}
                )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        # combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
        verbose=True
    )
    return conversation_chain


def handle_userinput(user_question):

    translator = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang='eng_Latn', tgt_lang='kor_Hang', max_length=512)

    # text = 'Lockheed Martin Delivers Initial 5G Testbed To U.S. Marine Corps And Begins Mobile Network Experimentation'
    # output = translator(text, max_length=512)
    # print(output[0]['translation_text'])
    
    if st.session_state.conversation is None:
        return
    response = st.session_state.conversation({'question': user_question})
    
    output = translator(response["answer"], max_length=512)
    print(output[0]['translation_text'])
    print("chat_history:",response['chat_history'])
    # output = translator(response, max_length=512)[0]['translation_text']

    st.session_state.chat_history = response['chat_history']
    chat_history_reversed = reversed(st.session_state.chat_history)

    st.write("Question: ", user_question)
    st.write(output[0]['translation_text'])

    for i, message in enumerate(chat_history_reversed):
        if i % 2 == 0:
            # st.write(bot_template.replace(
            #     "{{MSG}}", message.content), unsafe_allow_html=True)
            st.write(message.content)
        else:
            # st.write(user_template.replace(
            #     "{{MSG}}", message.content), unsafe_allow_html=True)
            st.write(message.content)

def main():
    load_dotenv()
    # st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.set_page_config(page_title="Chat with multiple PDFs")

    # st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
            st.session_state.conversation = None
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = None


    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat",key=reset_button_key)
    if reset_button:
        st.session_state.conversation = None
        st.session_state.chat_history = None
        
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store 
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain 
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__== '__main__':
    main()