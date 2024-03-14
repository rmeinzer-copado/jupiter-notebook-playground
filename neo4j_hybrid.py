from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Get env vars
url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# Neo4jVector Module will connect to Neo4j and create a vector and keyword (ToDo - find keyword index) indices if needed
# db = Neo4jVector.from_documents(
#     docs,
#     OpenAIEmbeddings(),
#     url=url,
#     username=username,
#     password=password,
#     # below is optional to create and to use keyword index
#     search_type="hybrid",
# )

index_name = "vector"  # default index name
keyword_index_name = "keyword"  # default keyword index name

store = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
    keyword_index_name=keyword_index_name,
    # below is optional, however proves to increase similarity score from ~.90 to 1
    search_type="hybrid",
)

query = "What did the president say about Ketanji Brown Jackson"
# docs_with_score = store.similarity_search_with_score(query, k=2) # find two most similar docs

# for doc, score in docs_with_score:
#     print("-" * 50)
#     print("Score: ", score)
#     print(doc.page_content)
#     print("-" * 50)

# create retriever to use with GPT for Q&A over an index
retriever = store.as_retriever()
retriever.get_relevant_documents(query)[0]

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

# stuff top-doc into prompt for GPT
chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), chain_type="stuff", retriever=retriever
)

print(chain(
    {"question": "What did the president say about Ketanji Brown Jackson"},
    return_only_outputs=True,
))