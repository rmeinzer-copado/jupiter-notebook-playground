from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv
import os

load_dotenv()

# Get env vars
diffbot_api_key = os.getenv('DIFFBOT_API_KEY')
url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

graph = Neo4jGraph(url=url, username=username, password=password)
diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)

# query = "Warren Buffett"
# raw_documents = WikipediaLoader(query=query).load()
# graph_documents = diffbot_nlp.convert_to_graph_documents(raw_documents)

# graph.add_graph_documents(graph_documents)
# graph.refresh_schema()

from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
    qa_llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    graph=graph,
    verbose=True,
)

print(chain.run("Which university did Warren Buffett attend?"))

# > Entering new GraphCypherQAChain chain...
# Generated Cypher:
# MATCH (p:Person {name: "Warren Buffett"})-[:EDUCATED_AT]->(o:Organization)
# RETURN o.name
# Full Context:
# [{'o.name': 'Alice Deal Junior High School'}, {'o.name': 'Columbia Business School'}, {'o.name': 'Woodrow Wilson High School'}]

# > Finished chain.
# Warren Buffett attended Columbia Business School.