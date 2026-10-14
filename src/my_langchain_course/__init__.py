import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print("Hello from my-langchain-course!")

    print(os.environ.get("GROQ_API_KEY"))


# uv run my-langchain-course
