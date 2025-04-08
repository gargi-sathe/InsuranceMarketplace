import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from langchain_community.document_loaders import PyPDFLoader
import re
import datetime
import requests
from groq import Groq  # Ensure 'groq' is installed (`pip install groq`)
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Updated import
# from langchain.schema import RunnableLambda
from langchain_core.runnables import RunnableParallel
import threading

from neo4j import GraphDatabase

GROQ_API_KEY = []
g_llm = OllamaLLM(model="gemma3:12b")
token_l=len(GROQ_API_KEY)
URI = "bolt://localhost:7687"  # Adjust if using a remote instance
USERNAME = "neo4j"
PASSWORD = "password"
groq_model_d="deepseek-r1-distill-llama-70b"
groq_model_l="llama3-70b-8192"
COUNT=0
TOKEN_LIMIT=4000
MAX_WORKERS = 10
GEMINI_API_KEY = ""  # Replace with your actual API key
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


counter_lock = threading.Lock()

def increment_counter():
    global COUNT
    with counter_lock:
        current = COUNT
        COUNT += 1
        return current


def parse_medicare_chapter(pdf_path, chapter_num):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Extract the specific chapter - MORE FLEXIBLE PATTERN
    chapter_pattern = re.compile(f"Chapter\\s*{chapter_num}[:\\s]", re.IGNORECASE)
    chapter_text = ""
    chapter_found = False

    for page in pages:
        if chapter_pattern.search(page.page_content):
            chapter_found = True
            chapter_text += page.page_content
        elif chapter_found and f"Chapter {chapter_num + 1}" not in page.page_content:
            # Continue adding content until we hit the next chapter
            chapter_text += page.page_content

    # Parse sections
    sections = []
    section_pattern = re.compile(r"Section (\d+\.\d+) (.*)")
    current_section = None

    for line in chapter_text.split('\n'):
        section_match = section_pattern.match(line)
        if section_match:
            if current_section:
                sections.append(current_section)
            section_num, section_title = section_match.groups()
            current_section = {
                "title": section_title,
                "content": ""
            }
        elif current_section:
            current_section["content"] += line + "\n"

    if current_section:
        sections.append(current_section)

    # Create chapter structure
    chapter = sections

    return chapter


def validate_json_structure(json_str_val):
    try:
        # Parse the JSON string
        data = json.loads(json_str_val)

        # Check if there's at least one top-level attribute
        if len(data) < 1:
            return True, "JSON should have at least one top-level attribute"

        # Iterate through all top-level attributes
        for main_attr_name, main_attr in data.items():
            # Check if main attribute is an object
            if not isinstance(main_attr, dict):
                return False, f"The main attribute '{main_attr_name}' should be an object"

            # Check if all values in the main attribute are UUID strings
            for sub_attr, value in main_attr.items():
                # Check if the value is a string
                if not isinstance(value, str):
                    return False, f"Value of '{sub_attr}' in '{main_attr_name}' is not a string"

        return True, "JSON structure is valid"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def add_text_summary( sub_attr_id, summary_name, text_content, properties=None):

    node_id = str(uuid.uuid4())
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:

        # Check if the string is a valid UUID
        uuid.UUID(sub_attr_id)
    except ValueError:
        print( f"Value of '{sub_attr_id}' is not a valid UUID:")
        return ""

    with driver.session() as session:
        # Create text summary and connect to sub-attribute
        cypher_query = (
            "MATCH (s:SubAttribute {id: $sub_id}) "
            "CREATE (t:TextSummary {id: $id, name: $name, content: $content})-[:DESCRIBES]->(s) "
            "RETURN t"
        )

        params = {
            "sub_id": sub_attr_id,
            "id": node_id,
            "name": summary_name,
            "content": text_content
        }

        # Add any additional properties
        if properties:
            property_set = ", ".join([f"t.{key} = ${key}" for key in properties.keys()])
            cypher_query = (
                "MATCH (s:SubAttribute {id: $sub_id}) "
                "CREATE (t:TextSummary {id: $id, name: $name, content: $content})-[:DESCRIBES]->(s) "
                f"SET {property_set} "
                "RETURN t"
            )
            params.update(properties)

        session.run(cypher_query, params)

    return node_id

def create_main_attribute(name, properties=None):

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

tdate = datetime.datetime.now()

# Connect to Neo4j and insert data
def insert_data(jsondata1, description, title):
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        for category, terms in jsondata1.items():
            if category not in data_attributes:
                mainattr_id=create_main_attribute(category, {"created_date": str(datetime.datetime.now())})
                data_main_attributes[category]=mainattr_id
                data_attributes[category]={}
            if isinstance(terms, str):
                add_text_summary( terms, title, description,{"created_date": str(datetime.datetime.now())})
            else:
                for term, desc in terms.items():
                    if data_attributes[category]=={} or (term not in data_attributes[category]):
                        mainattr_id=data_main_attributes[category]
                        subattr_id=create_sub_attribute(mainattr_id, term, {"created_date": str(datetime.datetime.now())})
                        data_attributes[category][term]=subattr_id
                        desc=subattr_id
                    add_text_summary(desc, title, description, {"created_date": str(datetime.datetime.now())})
    print("Nodes and relationships created successfully!")
    driver.close()


with open("data.json", "r") as file_1:
    data_attributes=json.load(file_1)


with open("main_attribute_data.json","r") as file_2:
    data_main_attributes=json.load(file_2)


def query_groq(g_prompt, groq_model):
    client = Groq(api_key=GROQ_API_KEY[increment_counter() % token_l])
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": g_prompt}],
        model=groq_model)
    return chat_completion.choices[0].message.content
def groq_summary(text, llm, txt_length):
    # Construct the question string
    your_question = (
        "Can you summarize the text in a short paragraph without missing any key points? "
        "Here is the text: "
    )
    full_prompt = f"{your_question}{text}"
    response=""
    # print(full_prompt)
    repeat_loop_s=False
    max_retry_s=3
    while not repeat_loop_s and max_retry_s>0:
        if txt_length<=TOKEN_LIMIT:
            try:
                req_response = query_groq(full_prompt, groq_model_l)
                response=req_response.split(":",1)[1]
                repeat_loop_s=True
            except:
                thread_id = threading.get_ident()
                thread_name = threading.current_thread().name
                print(f"Failed to process summery for groq the task with {thread_id} and {thread_name}")
                max_retry_s-=1
        else:
            # Define the prompt template
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Can you summarize the following text concisely without missing any key points? {text}"
            )
            # Execute the chain
            chain = prompt | llm
            # response = chain.invoke({"text": text})
            # Request payload
            payload = {
                "contents": [{
                    "parts": [{"text": your_question}]
                }]
            }

            # Headers
            headers = {
                "Content-Type": "application/json"
            }
            try:
                req_response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
                response = req_response.json()["candidates"][0]["content"]["parts"][0]["text"]
                repeat_loop_s = True
            except:
                thread_id = threading.get_ident()
                thread_name = threading.current_thread().name
                print(f"Failed to process summery for gemini the task with {thread_id} and {thread_name}")
                max_retry_s-=1


    # print(response)
    # print("\n\n")


    return response.replace("\n","")

def process_response(response_1):
    # print("line 340")
    # print(response_1)
    tag = "</think>"
    text = ""
    tag_position = response_1.find(tag)
    after=""
    # If the tag exists, return everything after the tag
    if tag_position != -1:
        text = response_1[tag_position + len(tag):]
    else:
        # If tag not found, return the original text
        text = response_1
    if text.find("{"):
        before, after = text.split("{", 1)
    after = "{" + after
    # with open("final_one_11.txt", "a") as file_3:
    #     file_3.write(after)

    after_rr = after.rstrip('`').strip()
    # print("after strip response:    ")
    # print(after_rr)
    # print("\n")
    valid_response_1, validation_reason = validate_json_structure(after_rr)
    if not valid_response_1:
        repeat_boolean_1 = True
    else:
        repeat_boolean_1 = False
    return after_rr, repeat_boolean_1, valid_response_1


def groq_relations(text, data, llm, txt_length):
    # Construct the question string efficiently
    global json_str, after_r
    repeat_boolean=False
    your_question = (
        "Given the following list of attributes:\n"
        f"{json.dumps(data)}\n\n"
        "Analyze the text below and categorize it using the attributes from the list above. Your task is to:\n"
        "1. Identify which main attributes and sub-attributes from the list best match the content\n"
        "2. Only create new attributes if no suitable match exists in the provided list\n"
        "3. Include the corresponding node IDs from the attribute list\n\n"
        "Return ONLY a JSON object in this exact format:\n"
        "{\n"
        " \"<main_attribute>\": {\n"
        " \"<sub_attribute_1>\": \"<node_id>\",\n"
        " \"<sub_attribute_2>\": \"<node_id>\"\n"
        " }\n"
        "}\n\n"
        f"TEXT TO ANALYZE: {text}\n\n"
        "IMPORTANT:\n"
        "- Return ONLY valid JSON with no explanation or additional text\n"
        "- Use attribute names from the provided list whenever possible\n"
        "- If nothing matches, return an empty JSON object: {}"
        )
    response=""
    valid_response=False
    max_retry=3
    while not valid_response and max_retry>0:
        if repeat_boolean:
            your_question = (
                "You are given a JSON object named AVAILABLE_ATTRIBUTES.\n\n"

                "AVAILABLE_ATTRIBUTES:\n"
                f"{json.dumps(after_r, indent=2)}\n\n"

                "Your task:\n"
                "- Restructure the above AVAILABLE_ATTRIBUTES into a hierarchical format.\n"
                "- Use ONLY main topics and their sub-topics based on how they are grouped or related in the input.\n"
                "- Each sub-topic must map to its corresponding node_id from the input.\n"
                "- DO NOT include any extra text, explanation, or raw input in the response — only output a well-formatted JSON.\n"

                "Output format example (strictly follow this structure):\n"
                "{\n"
                "  \"<Main Topic 1>\": {\n"
                "    \"<Sub-topic A>\": \"<node_id>\",\n"
                "    \"<Sub-topic B>\": \"<node_id>\"\n"
                "  },\n"
                "  \"<Main Topic 2>\": {\n"
                "    \"<Sub-topic C>\": \"<node_id>\"\n"
                "  }\n"
                "}\n\n"

                "Important:\n"
                "- Stick *exactly* to the structure above.\n"
                "- The final output must be valid JSON with correct nesting.\n"
                "- No bullet points, comments, or markdown — just pure JSON."
            )
            txt_length = len(str(your_question).split())
        # print(your_question)
        # print("repeat loop: ")
        # print(valid_response)
        # print(repeat_boolean)
        # print(your_question)
        # print("\n")

        if txt_length<=TOKEN_LIMIT:
            try:
                # print("line 438")
                response = query_groq(your_question, groq_model_d)
                # print("after response:  ")
                after_r,repeat_boolean, valid_response = process_response(response)
                max_retry-=1

            except:
                thread_id = threading.get_ident()
                thread_name = threading.current_thread().name
                print(f"failed to process the relation via groq with retry count: {max_retry} with thread id: {thread_id} thread name: {thread_name} \n")
                # print(response)
                valid_response=False
                repeat_boolean=False
                max_retry-=1
        else:
            # Create the prompt template
            prompt = PromptTemplate(
                input_variables=["question"],
                template="{question}"
            )

            # Execute the chain
            chain = prompt | llm
            try:
                # response = chain.invoke({"question": your_question.replace("\n","")})
                # Request payload
                payload = {
                    "contents": [{
                        "parts": [{"text": your_question}]
                    }]
                }

                # Headers
                headers = {
                    "Content-Type": "application/json"
                }

                response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
                raw_text_1 = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                after_r, repeat_boolean, valid_response = process_response(raw_text_1)
                max_retry-=1
            except :
                thread_id = threading.get_ident()
                thread_name = threading.current_thread().name
                print(f"failed to process the relation via gemini with retry count: {max_retry} with thread id: {thread_id} thread name: {thread_name} \n")
                valid_response = False
                repeat_boolean = False
                max_retry-=1
        # print("max retry: "+str(max_retry))

    return json.loads(after_r)

def section_execution(data, task_id_se, text_length, title):
    summery=groq_summary(data, g_llm, text_length)
    relation=groq_relations(data, data_attributes, g_llm, text_length)
    insert_data(relation, summery, title)
    thread_id = threading.get_ident()
    thread_name = threading.current_thread().name
    return f"Task {task_id_se} is completed with thread id: {thread_id} thread name: {thread_name}"

def count_words_in_dict(data):
    word_count = 0
    for key, value in data.items():
        word_count += len(str(key).split())
        word_count += len(str(value).split())
    return word_count

def process_section(section_1, task_id_1):
    l = count_words_in_dict(section_1)
    summary_title = section_1["title"]
    result = section_execution(section_1, task_id_1, l, summary_title)
    # print(f"Task {task_id_1} completed.")
    return result

for i in range(21):
    sections = parse_medicare_chapter("C:/UIC_COURSES/sem2/cs_532_nlp/project_insurence/EOC_1.pdf", i)
    print(f"Chapter {i}: {len(sections)} sections")

    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for task_id, section in enumerate(sections):
            future = executor.submit(process_section, section, task_id)
            futures.append(future)

        for future in as_completed(futures):
            r = future.result()
            print(r)  # or store it if needed

    # Save data after processing each chapter
    with open("data.json", "w") as file_1:
        json.dump(data_attributes, file_1)

    with open("main_attribute_data.json", "w") as file_2:
        json.dump(data_main_attributes, file_2)







