from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

chat = ChatGoogleGenerativeAI(
    
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# schema

class Review(TypedDict):

    summary: str
    sentiment: str

structureed_chat = chat.with_structured_output(Review)

result = structureed_chat.invoke ("""the hardware is great but the software feels bloated. There are too many pre installed apps that I can't remove . Also , the UI looks outdated compared to other brands . hoping for a software update to fix this.""")

print(result)