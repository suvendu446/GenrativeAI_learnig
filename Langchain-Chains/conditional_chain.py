import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Model
model_1 = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Schema
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Give the sentiment of the feedback"

    )

parser1=StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or negative:\n"
        "{feedback}\n"
        "{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

# Chain
classifier_chain = prompt1 | model_1 | parser2

prompt2 = PromptTemplate(
    template = "Write an appropiate responce to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3= PromptTemplate(
    template = "write an appropoate resposnce to this negative feedback \n {feedback}",
    input_variables= ['feedback']
)




# Run
#result = classifier_chain.invoke({"feedback": "This is a wonderful smartphone"}).sentiment

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model_1 | parser1 ),
    (lambda x:x.sentiment == "negative",prompt3 |model_1 | parser1),
    RunnableLambda(lambda x:"cound not find sentiments")

)

chain = classifier_chain | branch_chain

print(chain.send_message_stream({'feedback':'This is a nice phone'}))

chain.get_graph().print_ascii()