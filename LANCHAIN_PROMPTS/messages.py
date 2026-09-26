from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv

model=model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
   api_key=os.getenv("GEMINI_API_KEY")
)
Chat_history=[
    SystemMessage(content="you are helpfull chart assistant")
]

messages = [
    SystemMessage(content='you are helpfull assistant'),
    HumanMessage(content='tell me about langchain')
]

while True:
    user_input = input('you:')
    Chat_history.append(HumanMessage(content=user_input))
    if user_input =="exit":
        break
    result=model.invoke(Chat_history)
    Chat_history.append(AIMessage(content=result.content))
    print("AI: " ,result.content)

print(Chat_history)
