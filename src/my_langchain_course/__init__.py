from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-5-5")










def main() -> None:
    print("Hello from my-langchain-course!")


if __name__ == "__main__":
    main()