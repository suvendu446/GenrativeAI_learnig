from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch

load_dotenv()

prompt =PromptTemplate(
    template ="write a summery on {text}" ,
    input_variables = ["text"]

)

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

parser = StrOutputParser()

loder =TextLoader("cricket.txt")

docs = loder.load()

chain = prompt | model|parser

result = chain.invoke({'text':docs[0].page_content})

print(result)