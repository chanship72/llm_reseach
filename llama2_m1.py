from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain


template = """Question: {question}
Answer: Let's work this out in a step by step way to be sure we have the right answer."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = LlamaCpp(
	# model_path: 로컬머신에 다운로드 받은 모델의 위치
    model_path="./llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.0,
    top_p=1,
    max_tokens=8192,
    verbose=True,
    # n_ctx: 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
    n_ctx=4096 
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

prompt = """
Among Korean dramas, please recommend 3 medical dramas about hospital life. 
When recommending, classify it by number and title, and describe the release year and cast.
"""

response = llm_chain.run(prompt)
print(response)