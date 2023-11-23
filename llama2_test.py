
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp

n_gpu_layers = 8  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="llama-2-13b-chat.ggmlv3.q5_0.bin",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     n_ctx=2048,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
#     verbose=True,
# )
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
    model_type="llama",
    config = {'context_length' : 2048, 'gpu_layers' : 200}
)

# llm("Simulate a rap battle between Stephen Colbert and John Oliver")

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Build vectorstore and keep the metadata

from langchain.vectorstores import Chroma
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma(persist_directory="./data", embedding_function=embeddings)

# Run
question = "ESG"
docs = vectorstore.similarity_search(question)
result = llm_chain(docs)

# Output
print(result["text"])
# ['metadata']['source']
