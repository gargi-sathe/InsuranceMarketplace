import uuid
from datetime import datetime

from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

import json
import re
from neo4j import GraphDatabase

# Neo4j connection details
URI = "bolt://localhost:7687"  # Adjust if using a remote instance
USERNAME = "neo4j"
PASSWORD = "password"

# 1. Initialize the LLM (ensure Ollama is running and model is available)
llm = OllamaLLM(model="gemma3:12b")  # Change model name if needed

# 2. Load the PDF
pdf_path = "C:/UIC_COURSES/sem2/cs_532_nlp/project_insurence/EOC_1.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

if not documents:
    print("Error: No text extracted from the PDF.")
    exit()

# 3. Combine all text into a single string
text = "\n".join([doc.page_content for doc in documents])

# 4. Define the question or task
your_question = "Can you Extract Main attributes and each main attributes sub attributes to classify the content from the document and can you give me the output in json format?"

# 5. Define the prompt template
template = """
Based on the following document, answer the question below.

Document:
{text}

Question:
{question}

format:
{{
    "Main atttribute 1": {{
        "subattribute 1.1": "",
        "subattribute 1.2": ""
    }},
    "Main Attribute 2": {{
        "subattribute 2.1": "",
        "subattribute 2.2": ""
    }}
}}

Respond in JSON format only.
"""


prompt = PromptTemplate(
    input_variables=["text", "question"],
    template=template
)

# 6. Create the chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# 7. Run the chain with input values
response = llm_chain.invoke({
    "text": text,
    "question": your_question
})


raw_text = response["text"]

# Step 2: Remove the triple backticks and "json" tag
cleaned_text = re.sub(r"```json|```", "", raw_text).strip()

# Step 3: Parse to JSON
parsed_data = json.loads(cleaned_text)
print(parsed_data)
# 8. Save the output to a file
with open("data.json", "w") as f:
    json.dump(parsed_data, f, indent=2)

print("Output written to data.json")
URI = "bolt://localhost:7687"  # Adjust if using a remote instance
USERNAME = "neo4j"
PASSWORD = "password"
# Function to create nodes in Neo4j
def create_nodes(tx, category, term, description):
    query = """
    MERGE (c:Category {name: $category})
    MERGE (t:Term {name: $term, description: $description})
    MERGE (c)-[:CONTAINS]->(t)
    """
    tx.run(query, category=category, term=term, description=description)

def create_main_attribute(name, properties=None):
        """
        Create a main attribute node

        Args:
            name (str): Name of the main attribute
            properties (dict, optional): Additional properties for the node

        Returns:
            str: ID of the created node
        """
        node_id = str(uuid.uuid4())
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        with driver.session() as session:
            # Create main attribute node
            cypher_query = (
                "CREATE (m:MainAttribute {id: $id, name: $name}) "
                "RETURN m"
            )

            params = {"id": node_id, "name": name}

            # Add any additional properties
            if properties:
                property_set = ", ".join([f"m.{key} = ${key}" for key in properties.keys()])
                cypher_query = (
                    f"CREATE (m:MainAttribute {{id: $id, name: $name}}) "
                    f"SET {property_set} "
                    f"RETURN m"
                )
                params.update(properties)

            session.run(cypher_query, params)

        return node_id


def create_sub_attribute(main_attr_id, name, properties=None):
        """
        Create a sub-attribute node connected to a main attribute

        Args:
            main_attr_id (str): ID of the main attribute
            name (str): Name of the sub-attribute
            properties (dict, optional): Additional properties for the node

        Returns:
            str: ID of the created sub-attribute node
        """
        node_id = str(uuid.uuid4())
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        with driver.session() as session:
            # Create sub-attribute and connect to main attribute
            cypher_query = (
                "MATCH (m:MainAttribute {id: $main_id}) "
                "CREATE (s:SubAttribute {id: $id, name: $name})-[:BELONGS_TO]->(m) "
                "RETURN s"
            )

            params = {"main_id": main_attr_id, "id": node_id, "name": name}

            # Add any additional properties
            if properties:
                property_set = ", ".join([f"s.{key} = ${key}" for key in properties.keys()])
                cypher_query = (
                    "MATCH (m:MainAttribute {id: $main_id}) "
                    "CREATE (s:SubAttribute {id: $id, name: $name})-[:BELONGS_TO]->(m) "
                    f"SET {property_set} "
                    "RETURN s"
                )
                params.update(properties)

            session.run(cypher_query, params)

        return node_id

tdate = datetime.now()
mainattr_ids_json={}
# Connect to Neo4j and insert data
def insert_data(jsondata1):
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        for category, terms in jsondata1.items():
            category_id=create_main_attribute(category, {"created_date": str(tdate)})
            mainattr_ids_json[category]=category_id
            if isinstance(terms, str):
                sub_id=create_sub_attribute(category_id, terms, {"created_date": str(tdate)})
                jsondata1[category]=sub_id
            for term, description in terms.items():
                sub_id=create_sub_attribute(category_id, term, {"created_date": str(tdate)})
                terms[term]=sub_id
    print("✅ Nodes and relationships created successfully!")
    return jsondata1
    driver.close()

final_json=insert_data(parsed_data)

with open("data.json", "w") as f:
    json.dump(final_json, f, indent=2)

with open("main_attribute_data.json", "w") as f2:
    json.dump(mainattr_ids_json, f2)





