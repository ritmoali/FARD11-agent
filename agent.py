from langchain.llms import ollama
from langchain.agents import load_tools, initialize_agent
from langchain.agents.agent_types import AgentType

llm = ollama(model="llama3")
tools = load_tools(["llm_math"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = agent.run(user_input)
    print(f"🤖 FARDAI: {response}")