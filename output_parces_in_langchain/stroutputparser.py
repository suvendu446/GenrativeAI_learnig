from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# First prompt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

# Second prompt -> summarize the report
template2 = PromptTemplate(
    template="summarize the report on text.\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | llm | template2 | llm | parser

result=chain.invoke({"topic": "the impact of climate change on agriculture"})

print(result)
