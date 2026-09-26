from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",   
    api_key=os.getenv("GEMINI_API_KEY")
)
result = llm.invoke("what is the capital of india")

print(result.text)