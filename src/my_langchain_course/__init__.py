from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from typing import List
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy


# #custom tool

#from tavily import TavilyClient
# tavily_client = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query (str): the query to search for
#     Returns:
#         str: the search result
#     """
#     print(f"Searching for: {query}")
#     #return "weather in Tokyo is sunny and 25 degrees Celsius" #when it was a static response
#     #return tavily_client.search(query=query, max_results=2) #when we use TavilyClient to search over the internet
    
#     #to not hit a maximum
#     results = tavily_client.search(query=query, max_results=5)  # CHANGED: 2 -> 5, more jobs per search
#     return "\n".join(
#             f"{r['title']} - {r['url']}: {r['content'][:200]}"
#             for r in results["results"]
#     )

#The built in Tool
from langchain_tavily import TavilySearch


class Source (BaseModel):
    """schema for a source used by the agent"""    

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """schema for the agent response with answer and sources"""

    answer: str = Field(description="The content of the response to the query")
    sources: List[Source] = Field(default_factory=list, description="The sources used by the agent")


################
llm = ChatAnthropic(model="claude-haiku-5-5")
tools = [TavilySearch()]  

agent = create_agent(
    model=llm,
    tools=[TavilySearch(max_results=3)],
    response_format=AgentResponse,
)
################


def main() -> None:
    print("Hello from my-langchain-course!")
    #result = agent.invoke({"messages": [HumanMessage(content="What is the weather of Tokyo?")]})
    #we re sending a human message wih the content "What is the capital of Japan?"
    #it must be a list []
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 AI engineer job postings using LangChain in Germany and list the details")]}
    )
    #print(result["messages"][-1].content) #print the last message

    #the structured response
#   So agent.invoke(...) returns a dictionary that LangChain builds:    
#     result = {
#     "messages": [...],                         # always added by LangChain
#     "structured_response": AgentResponse(...)  # added because you used response_format
# }
    response = result["structured_response"]
    print(response.answer)
    print("\nSources:")
    for source in response.sources:
        print(source.url)
#or 
    #for message in result["messages"]:
    #    print(message.content)
if __name__ == "__main__":
    main()


#uv run my-langchain-course
