from langchain_core.messages import HumanMessage
from agent import agent

from dotenv import load_dotenv
_ = load_dotenv()

messages = [HumanMessage(content="I need low fat foods in Washington D.C.")]
result = agent.graph.invoke({"messages": messages})
json_date = result['json_output']
print(json_date)
