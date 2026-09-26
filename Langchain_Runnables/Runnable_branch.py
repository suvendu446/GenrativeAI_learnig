from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch
import os

load_dotenv ()

prompt_1 = PromptTemplate(
    template="Write a detaild topic about {topic}",
    input_variables =["topic"]

)

prompt_2 = PromptTemplate(
    template ="give a summary of text {text}",
    input_variables = ["text"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

parser = StrOutputParser()

Text_gen_chain = RunnableSequence(prompt_1,model,parser)

branch_chain = RunnableBranch(
    (lambda X: len(X.split())>300,RunnableSequence(prompt_2,model,parser)) , RunnablePassthrough()

)

final_chain = RunnableSequence(Text_gen_chain,branch_chain)

result =final_chain.invoke({"topic":"india vs china"})

print(result)