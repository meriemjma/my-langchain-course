from dotenv import load_dotenv
from langchain.chat_models import init_chat_model #helps us initialize the chat model with just a sting
from langsmith import traceable
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage #message containing the tool result


load_dotenv()  # Load environment variables from .env file

MAX_iTERATIONS = 10  # Maximum number of iterations for the agent loop
MODEL = "claude-haiku-5-5"  # Model to be used for the agent loop

# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product_name: str) -> float:
    """
    Get the price of a product.
    """
    product_prices = {
        "laptop": 999,
        "smartphone": 699,
        "headphones": 199,
    }
    print(f"Getting price for product: {product_name}")
    return product_prices.get(product_name.lower(), "Product not found.")

# the doc strings and the types are very important because it's going to be passed to the LLM
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price based on the discount tier.
    """
    discount_tiers = {
        "silver": 10,
        "gold": 20,
        "platinum": 30,
    }
    discount = discount_tiers.get(discount_tier.lower(), 0)
    print(f"Applying discount: {discount}%")
    discounted_price = price * (1 - discount / 100)
    return discounted_price

@traceable (name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    #This builds a dictionary where each tool’s name points to the tool itself:
    #     tools_dict = {
    #     "get_product_price": get_product_price,
    #     "apply_discount": apply_discount,
    # }
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"anthropic:{MODEL}")

    #we want to let the model know which tools we have now
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 50)

    #List of messages that will be passed to the LLM
    messages = [
        SystemMessage(
            content=("You are a helpful shopping assistant. "
            "You have access to a product price tool and a discount tool."
            )
        ),
        HumanMessage(content=question),
    ]  

    for iteration in range(MAX_iTERATIONS):
        print(f"Iteration {iteration + 1}/{MAX_iTERATIONS}")

        ai_response = llm_with_tools.invoke(messages)
        # ai_response
        #     ├── content    = ""
        #     └── tool_calls = [ get_product_price("laptop") ]   ← this part
        tool_calls= ai_response.tool_calls

        #if no tool calls, this is the final answer
        if  not tool_calls:
            print("Final Answer:", ai_response.content)
            return ai_response.content
        tool_call = tool_calls[0]  # Assuming only one tool call per iteration


        # tool_call = {
        #                 "name": "get_product_price",           # which tool
        #                 "args": {"product_name": "laptop"},    # with what inputs
        #                 "id": "toolu_123",                     # a unique ID
        #             }
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {}) #prouct_name="laptop" or price=999, discount_tier="gold"
        tool_call_id = tool_call.get("id")

        print(f"Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            print(f"Tool {tool_name} not found.")
            return "Error: Tool not found."

        #tool_args is {"product_name": "laptop"}
        #get_product_price("laptop")
        observation = tool_to_use.invoke(tool_args)
        #[Tool result]: 999
        print(f"[Tool result]: {observation}")

        #feeding back all the steps
        messages.append(ai_response)  # Add the AI's response to the messages
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Running agent loop...")
    result = run_agent("What is the price of a laptop and apply a gold discount?")