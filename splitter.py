import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

class translator(object):
    def __init__(self, src_lang, tgt_lang):
        self.model = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def translate(self, input_text):
        return self.model(input_text, max_length=1024)


with open('data/stt.dat', 'rt', encoding='UTF8') as fp:
    stt = fp.read()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len
    )
    return text_splitter.create_documents([stt])

# texts = text_splitter.create_documents([stt])

chunks = get_text_chunks(stt)
# print(chunks)

def get_vectorstore(text_chunks):    
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = get_vectorstore(chunks)
retriever = vectorstore.as_retriever(    
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
    )

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama")

query = "클래시파이어는 뭐가 있어?"

# retrieved_docs = retriever.invoke(query)
# for i in range(len(retrieved_docs)):
#     print(retrieved_docs[i].page_content)

# qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(),
# )

# print(qa_chain(query))

prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# Run
# question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(query)
# print(docs)

translated_docs = translator("kor_Hang", "eng_Latn").translate(str(docs)[:1024])
translated_docs = translated_docs[0]['translation_text']
# print(translated_docs[:512])
# print(docs)
result = llm_chain(translated_docs[:512])

# Output
print(result['docs'])
print(translator("eng_Latn", "kor_Hang").translate(result['text'])[0]['translation_text'])