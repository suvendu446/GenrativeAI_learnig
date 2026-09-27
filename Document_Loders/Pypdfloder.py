from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch

#load_dotenv

loader = PyPDFLoader("machine_learning_basics.pdf")

docs=loader.load()


print(docs)

print(docs[0].page_content)