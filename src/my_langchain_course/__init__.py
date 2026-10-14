from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


                
def main() -> None:
    print("Hello from my-langchain-course!")
    #print(os.environ.get("GROQ_API_KEY"))
    information = """
                Marie Curie (born Maria Skłodowska on November 7, 1867, in Warsaw, Poland) was a physicist and chemist known for her pioneering research on radioactivity.
                She moved to Paris in 1891 to study at the Sorbonne, where she earned degrees in physics and mathematics. There she met Pierre Curie, whom she married in 1895.
                Together they discovered two new elements, polonium and radium. In 1903, Marie and Pierre Curie shared the Nobel Prize in Physics with Henri Becquerel for their work on radioactivity.
                In 1911, she won a second Nobel Prize, this time in Chemistry, making her the first person to win Nobel Prizes in two different sciences.
                During World War I, she developed mobile X-ray units, known as "little Curies," to help doctors treat wounded soldiers.
                She died on July 4, 1934, from an illness linked to her long exposure to radiation.
                """

    summary_template = """
    given the infomation {information} about a person i want you to create:
    1. A short summary of the person in 3 sentences.
    2. two interesting facts about the person.
    """

    summary_prompt = PromptTemplate(input_variables=["information"], template= summary_template)
    #Alternaive
    #summary_prompt = PromptTemplate("information", summary_template)
    llm = ChatGroq(temperature=0, model="openai/gpt-oss-120b")
    #temperature =0 : determistic 
    #temperature =1 : creative

    #LCEL: Langchain expression language: we create a chain by composing two componnents
    #the pipe | is creating a new runnable by connecting the output of the left component as in put to the right component
    #so this chain means we first want to fromat the input using the promt template and then we want to pass the output of the prompt template to the llm to get the final result
    #The resulting chain is called a runnable object.
    chain = summary_prompt | llm
    response = chain.invoke({"information": information}) #we plugging the information
    print(response.content)

if __name__ == "__main__":
    main()
# uv run my-langchain-course
