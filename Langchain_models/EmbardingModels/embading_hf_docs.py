from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents =[
     "delhi is the capital of india",
     "kolkata is the capital of new delhi",
     'paris is the capital of france'
]

vector = embedding.embed_documents(documents)

print(str(vector))