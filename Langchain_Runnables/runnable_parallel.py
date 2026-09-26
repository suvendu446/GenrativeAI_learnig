from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence , RunnableParallel
import os

load_dotenv()

prompt_1 = PromptTemplate(
    template = "generate a tweet about {topic}",
    input_variables = ["topic"]
)

prompt_2 = PromptTemplate(
    template = "Generate a linnked in post about {topic}",
    input_variables =["topic"]
)
model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = os.getenv("GIMINI_API_KEY")

)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet':RunnableSequence(prompt_1,model,parser),
        'post':RunnableSequence(prompt_2,model,parser)
    }
)

result = parallel_chain.invoke({"topic":"ai"})

print(result["tweet"])
print(result['post'])

