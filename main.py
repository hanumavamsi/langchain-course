from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tavily import TavilyClient

tavily = TavilyClient()


# Define a custom search tool using the @tool decorator
@tool
def search(query: str) -> str:  # Args and Returns needed for @tool decorator
    """
    A search tool that searches the web.
    Args :
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")

    response = tavily.search(query=query, num_results=1)

    return response


class Source(BaseModel):
    """A source returned by the search tool."""

    url: str = Field(..., description="The URL of the source")


class searchResponse(BaseModel):
    """The response returned by the search tool."""

    results: str = Field(..., description="The search results")
    sources: list[Source] = Field(
        default_factory=list, description="The sources of the search results"
    )


llm = ChatOpenAI(model="gpt-5")
# llm = ChatOllama(model="sparksammy/tinysam-l3.2", temperature=0) # Example of using Ollama LLM which has tools enabled by default

tools = [TavilySearch()]  # or use [search] to use the custom tool defined above
agent = create_agent(model=llm, tools=tools, response_format=searchResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
