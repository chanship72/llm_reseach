import os
import re

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import GPT2TokenizerFast
from transformers import pipeline

class translator(object):
    def __init__(self, src_lang, tgt_lang):
        self.model = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def translate(self, input_text):
        return self.model(input_text, max_length=1024)
    
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

with open('data/stt.dat', 'rt', encoding='UTF8') as fp:
    stt = fp.read()

def get_text_chunks(input_text):
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size = 300,
        chunk_overlap  = 0
    )
    # return text_splitter.split_text(text)
    return text_splitter.create_documents([input_text])

# texts = text_splitter.create_documents([stt])
kor_stt = re.sub(r"[^ㄱ-ㅣ가-힣\s]", "", stt)
# kor_stt = kor_stt.replace("\n","")
kor_stt = kor_stt.replace("\t","")

with open('data/stt_kor.dat', 'w', encoding='UTF8') as f:
    f.write(kor_stt)

# print(kor_stt)

kor_stt_chunks = get_text_chunks(kor_stt)
print(kor_stt_chunks)
print("size:",len(kor_stt_chunks))
# print(kor_stt_chunks[0])
# print(kor_stt_chunks[0].page_content)
total_str = ""
for i in range(len(kor_stt_chunks)):
    print(kor_stt_chunks[i].page_content)
    translated_stt = translator("kor_Hang", "eng_Latn").translate(kor_stt_chunks[i].page_content)
    print(translated_stt)

    translated_stt = translated_stt[0]['translation_text']
    print(translated_stt)

    total_str = total_str + translated_stt

print(total_str)

# with open('data/stt_eng.dat', 'w', encoding='UTF8') as f:
#     f.write(translated_stt)

# chunks = get_text_chunks(stt)
# # print(chunks)

# def get_vectorstore(text_chunks):    
#     # embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = Chroma.from_documents(chunks, embeddings)
#     return vectorstore

# vectorstore = get_vectorstore(chunks)
# retriever = vectorstore.as_retriever(    
#     search_type="mmr",  # Also test "similarity"
#     search_kwargs={"k": 8},
#     )

# llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama")
# query = "이 강의의 주제가 뭔가요?"

# # retrieved_docs = retriever.invoke(query)
# # for i in range(len(retrieved_docs)):
# #     print(retrieved_docs[i].page_content)

# # qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
# # qa_chain = RetrievalQA.from_chain_type(
# #     llm,
# #     retriever=vectorstore.as_retriever(),
# # )

# # print(qa_chain(query))

# prompt = PromptTemplate.from_template(
#     "Summarize the main themes in these retrieved docs: {docs}"
# )

# # Chain
# llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# # Run
# # question = "What are the approaches to Task Decomposition?"
# docs = vectorstore.similarity_search(query)
# # print(docs)

# translated_docs = translator("kor_Hang", "eng_Latn").translate(str(docs)[:1024])
# translated_docs = translated_docs[0]['translation_text']
# # print(translated_docs[:512])
# # print(docs)
# result = llm_chain(translated_docs[:512])

# # Output
# print(result['docs'])
# print(translator("eng_Latn", "kor_Hang").translate(result['text'])[0]['translation_text'])