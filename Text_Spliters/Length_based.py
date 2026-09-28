from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("machine_learning_basics.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=" "
)

result = splitter.split_documents(docs)

print(result[0].page_content)   # first chunk
