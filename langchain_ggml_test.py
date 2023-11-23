# from ctransformers import AutoModelForCausalLM
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id = "TheBloke/Llama-2-13B-chat-GGML"


# check ctransformers doc for more configs
config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 
          'temperature': 0.1, 'stream': True}

# llm = AutoModelForCausalLM.from_pretrained(
#       model_id, 
#       model_type="llama",                                           
#       #lib='avx2', for cpu use
#       gpu_layers=130, #110 for 7b, 130 for 13b
#       **config
#       )
# llm = CTransformers(model=model_id, model_type="llama", config=config, gpu_layers=130, callbacks=[StreamingStdOutCallbackHandler()])

# streaming
# llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", gpu_layers=130, callbacks=[StreamingStdOutCallbackHandler()])
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", gpu_layers=130)
# prompt="""Write a poem to help me remember the first 10 elements on the periodic table, giving each
# element its own line."""

# pt-br for multilingual eval
prompt2="""재무제표 분석의 핵심은 무엇인가요?"""

# tokens = llm.tokenize(prompt)
# print(llm(prompt, stream=False))

kr_en_translator = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang='kor_Hang', tgt_lang='eng_Latn', max_length=512)
kr_en_output = kr_en_translator(prompt2, max_length=512)[0]['translation_text']
print("translated_prompt:", kr_en_output)

result = llm(kr_en_output, stream=False)

print("result:",result)

en_kr_translator = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang='eng_Latn', tgt_lang='kor_Hang', max_length=512)

# # text = 'Lockheed Martin Delivers Initial 5G Testbed To U.S. Marine Corps And Begins Mobile Network Experimentation'

output = en_kr_translator(result, max_length=512)
print("translated:",output[0]['translation_text'])