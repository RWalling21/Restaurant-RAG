from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Any
import operator

from src.tools import tools 
from src.restaurant import RestaurantJSON

from dotenv import load_dotenv
_ = load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    json_output: Any

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("output-parser", self.parse_output_to_string)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: "output-parser"}
        )
        graph.add_edge("action", "llm")
        graph.add_edge("output-parser", END)
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def parse_output_to_string(self, state: AgentState):
        # To be clear, when dereferencing state['messages][-1].content we are not only returning the last message, we are returning a list of all messages because of the way that the messages variable is Annotated
        message_content = str(state['messages'][-1].content)
        json_output = JsonOutputParser(pydantic_object=RestaurantJSON).invoke(message_content)
        return {'json_output': json_output}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling the {t['name']} with args {t['args']}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Finished tool cycle")
        return {'messages': results}

prompt = """You are an AI assistant tasked with scraping web data for database use. 
Your specific duties involve collecting detailed information about restaurants, including their names, phone numbers, emails, dietary options, and customer reviews.

Follow these guidelines: 
- Utilize the search engine to begin finding information, the search tool is expensive so use it sparingly. 
- Utilize the read-tool to read links found with the search tool. 
- Make search and read calls as needed, but find the information with the least number of calls possible. 
- If preliminary information is needed for a follow-up query, obtain it before proceeding with further questions. 
- Once data collection is complete, compile and return the information exclusively in JSON format. Only return the JSON output.
- Try to create as many JSON objects as possible, but keep the number of tool calls low. 
- All data MUST BE REAL, only include information that has been directly sited from a review site / restaurant website / restaurant blog. 

Expected JSON output format:
```json
{
  "restaurants": [
    {
      "name": "Example Restaurant",
      "phone_number": "000-000-0000",
      "email": "Example Email",
      "dietary_options": ["vegan", "nut-free"],
      "customer_reviews": [
        {
            "reviewer": "name",
            "review": "review text"
        }
    ]
    }
  ]
}
"""

model = ChatOpenAI(model="gpt-4o")
agent = Agent(model, tools, system=prompt)
