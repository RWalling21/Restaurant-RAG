from langchain_core.messages import HumanMessage
from agent import agent
from dotenv import load_dotenv
import json

_ = load_dotenv()

if __name__ == "__main__":
    user_input = input("Please enter your query: ")
    messages = [HumanMessage(content=user_input)]

    result = agent.graph.invoke({"messages": messages})

    json_string = json.dumps(result['json_output'], indent=2)
    print(json_string)
