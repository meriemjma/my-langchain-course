from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


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

    
################
llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0,max_tokens=2000) 
tools = [TavilySearch()] 

agent = create_agent(
    model=llm,
    tools=tools,
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
    print(result["messages"][-1].content) #print the last message
#or 
    #for message in result["messages"]:
    #    print(message.content)
if __name__ == "__main__":
    main()


#uv run my-langchain-course
