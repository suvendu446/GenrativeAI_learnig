from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch
import os



load_dotenv()

print("USER_AGENT:", os.getenv("USER_AGENT"))

url="https://www.manati.in/"

loder=WebBaseLoader(url)

docs = loder.load()

print("Documents:", len(docs))
print(docs[0].page_content[:500])

Prompt= PromptTemplate(
    template = "Answer the {question} about the following {topic}",
    input_variables = ['question','topic']

)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

parser = StrOutputParser()

chain = Prompt | model | parser

result=chain.invoke({'question':"in which field manati working ?","topic":docs[0].page_content})

print(result)