from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# First prompt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

# Second prompt -> summarize the report
template2 = PromptTemplate(
    template="summarize the report on text.\n{text}",
    input_variables=["text"]
)

# Generate detailed report
prompt1 = template1.format_prompt(
    topic="Climate change and its impact on global agriculture"
).to_string()

result1 = llm.invoke(prompt1)
print("Detailed Report:\n", result1.content)

# Summarize the report
prompt2 = template2.format_prompt(text=result1.content).to_string()
result2 = llm.invoke(prompt2)
print("\nSummary:\n", result2.content)

