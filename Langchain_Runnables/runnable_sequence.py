from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="AFC is enabled with max remote calls",
)

load_dotenv()

# Create first prompt

prompt_1 = PromptTemplate(
    template ="write a joke on {topic}",
    input_variables=['topic']
)

#define the model

model = ChatGoogleGenerativeAI(
    model= 'gemini-3.6-flash',
    google_api_key = os.getenv("GIMINI_API_KEY")
)

#define a parser

parser = StrOutputParser()

prompt_2 = PromptTemplate(
    template = "explain the joke{text}",
    input_variables = ['text']
)

#build a chain

chain = RunnableSequence(
    prompt_1,model,parser,prompt_2,model,parser
)

result = chain.invoke({'topic':'ai'})

print(result)