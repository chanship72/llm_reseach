from ctransformers import AutoModelForCausalLM

model_id = "TheBloke/Llama-2-13B-chat-GGML"


# check ctransformers doc for more configs
config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 
          'temperature': 0.1, 'stream': True}

llm = AutoModelForCausalLM.from_pretrained(
      model_id, 
      model_type="llama",                                           
      #lib='avx2', for cpu use
      gpu_layers=130, #110 for 7b, 130 for 13b
      **config
      )

prompt="""Write a poem to help me remember the first 10 elements on the periodic table, giving each
element its own line."""

# pt-br for multilingual eval
prompt2="""Quando e por quem o Brasil foi descoberto?"""

# tokens = llm.tokenize(prompt)
print(llm(prompt, stream=False))