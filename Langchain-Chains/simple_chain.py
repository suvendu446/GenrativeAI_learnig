import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser  # Fixed here
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt = PromptTemplate(
    template="generate 5 interesting facts about {topic}",
    input_variables=["topic"],
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

parser = StrOutputParser()  # Fixed here

chain = prompt | model | parser

result = chain.invoke({"topic": "cricket"})

print(result)

chain.get_graph().print_ascii()