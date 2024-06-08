from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import Json
from typing import TypedDict, Annotated, Any
import operator

from tools import tools, RestaurantJSON

from dotenv import load_dotenv
_ = load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    json_output: Json[Any]

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
        message_content = str(state['messages'][-1].content)
        json_output = JsonOutputParser(pydantic_object=RestaurantJSON).invoke(message_content)
        return {'json_output': json_output}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """You are an AI assistant tasked with scraping web data for database use. 
Your specific duties involve collecting detailed information about restaurants, including their names, phone numbers, emails, dietary options, and customer reviews.

Follow these guidelines: 
- Utilize the search engine to identify relevant web pages and begin to gather information. 
- Employ the read-tool to dig deaper and read links mentioned from the seach tool. 
- Make multiple search and read-tool calls as needed, either concurrently or sequentially. 
- Initiate searches only when the required information parameters are clearly defined. 
- If preliminary information is needed for a follow-up query, obtain it before proceeding with further questions. 
- Once data collection is complete, compile and return the information exclusively in JSON format. Interaction with the user is not required beyond providing the JSON output.
- Collect enough information to make at least 5 complete JSON objects.
- All data MUST BE REAL, only include information that has been directly sited from a review site / restaurant website. 
"""

model = ChatOpenAI(model="gpt-4o")
agent = Agent(model, tools, system=prompt)
