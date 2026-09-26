from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage , HumanMessage

chat_template = ChatPromptTemplate([
   ('system',"you are a heplpful {domain} expert"),
   ('human',"explain in simple terms , what is {topic}")

])

prompt = chat_template.invoke({'domain':'cricket','topic':'dusra'})

print(prompt)