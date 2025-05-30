from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

# Load LLaMA 3 model via Ollama
llm = OllamaLLM(model="llama3")

# Load tools (e.g., math tool)
tools = load_tools(["llm-math"], llm=llm)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Chat loop
while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = agent.run(user_input)
    print(f"🤖 FARDAI: {response}")
