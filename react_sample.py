from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.llms import CTransformers
from langchain.docstore import Wikipedia
from langchain.llms import OpenAI

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]

# llm = OpenAI(temperature=0, model_name="text-davinci-002")
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama")
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
react.run(question)