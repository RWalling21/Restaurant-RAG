from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
import requests

search_url = "https://s.jina.ai/"
read_url = "https://r.jina.ai/"

class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class ReadInput(BaseModel):
    url: str = Field(description="should be a website URL")

@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search_tool(query: str) -> str:
    """Search a query online and output the markdown, this is an expensive operation use only when needed"""
    return requests.get(search_url + query)

@tool("read-tool", args_schema=ReadInput, return_direct=True)
def read_tool(url: str) -> str:
    """Return markdown of given URL, less computationally expensive than the search tool"""
    return requests.get(read_url + url)

tools = [search_tool, read_tool]
