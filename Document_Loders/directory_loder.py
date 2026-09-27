from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=r"C:\Users\suven\OneDrive\Desktop\Notes",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# lazy_load returns a generator
docs = loader.lazy_load()

# Print only the first page (first yielded document)
first_doc = next(docs)   # get the first item from generator
print(first_doc.page_content)
