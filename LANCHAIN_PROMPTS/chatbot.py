from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("🤖 Chatbot initialized! Type 'exit' to quit.\n")

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        break

    if not user_input.strip():
        continue

    try:
        # Add user message
        chat_history.append(
            HumanMessage(content=user_input)
        )

        # Send conversation history
        result = model.invoke(chat_history)

        # Extract only text
        ai_response = result.content[0]["text"]

        # Add only clean text to history
        chat_history.append(
            AIMessage(content=ai_response)
        )

        print("AI:", ai_response)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}\n")

print("\nChat History:")

for message in chat_history:
    print(f"{message.__class__.__name__}: {message.content}")