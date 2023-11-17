import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

translator = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang='eng_Latn', tgt_lang='kor_Hang', max_length=512)

text = 'Lockheed Martin Delivers Initial 5G Testbed To U.S. Marine Corps And Begins Mobile Network Experimentation'

output = translator(text, max_length=512)
print(output[0]['translation_text'])