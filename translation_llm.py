import torch
import logging

from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S')

# INFO 레벨 이상의 로그를 콘솔에 출력하는 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class translator(object):
    def __init__(self, src_lang, tgt_lang):
        # self.model = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", gpu_layers=130)
        # self.model = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang='kor_Hang', tgt_lang='eng_Latn', max_length=512)
        self.model = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def translate(self, input_text):
        return self.model(input_text, max_length=1024)

class LLM_model(object):
    def __init__(self, id, model_type):
        self.model = CTransformers(model=id, model_type=model_type, gpu_layers=130)

    def completion(self, input_text):
        return self.model(input_text, stream=False)

# pt-br for multilingual eval
prompt="재무제표 분석의 핵심은 무엇인가요?"
logger.info(prompt)

translated_prompt = translator("kor_Hang", "eng_Latn").translate(prompt)
translated_prompt = translated_prompt[0]['translation_text']
logger.info(translated_prompt)

llm_response = LLM_model(id="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama").completion(translated_prompt)
logger.info(llm_response)

translated_answer = translator("eng_Latn", "kor_Hang").translate(llm_response)
translated_answer = translated_answer[0]['translation_text']
logger.info(translated_answer)
