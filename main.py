from langchain_core.messages import HumanMessage
from agent import agent
from dotenv import load_dotenv
import json

_ = load_dotenv()

if __name__ == "__main__":
    user_input = input("Please enter your query: ")
    messages = [HumanMessage(content=user_input)]

    result = agent.graph.invoke({"messages": messages})

    json_data = json.loads(result['json_output'][-1])
    json_string = json.dumps(json_data, indent=2)
    print(json_string)
