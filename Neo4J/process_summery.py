import json

from neo4j import GraphDatabase
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

GROQ_API_KEY = ["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","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","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","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","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","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","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","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","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","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","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","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"]
g_llm = OllamaLLM(model="gemma3:12b")
token_l=len(GROQ_API_KEY)
URI = "bolt://localhost:7687"  # Adjust if using a remote instance
USERNAME = "neo4j"
PASSWORD = "password"
groq_model_d="deepseek-r1-distill-llama-70b"
groq_model_l="llama3-70b-8192"
# global count
count=0
TOKEN_LIMIT=3800
GEMINI_API_KEY = ["AIzaSyAxJzTmQfKujg8x7Tq-0ALLN-kicSmmd8M","AIzaSyBGiXOgo7MqY2Z86C2hU0H3y1sSQPh-Ywk","AIzaSyCIiKHR_ridCwCoPVKP-AYixLvW4UCghZs","AIzaSyBu-e2Uj1EShfvx3LDzKuCP5_aA7gPoIJk"]  # Replace with your actual API key
g_l=len(GEMINI_API_KEY)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="

with open("data.json", "r") as file_1:
    data_attributes=json.load(file_1)

def query_groq(g_prompt, groq_model, count_g=0):
    client = Groq(api_key=GROQ_API_KEY[count_g % token_l])
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": g_prompt}],
        model=groq_model)
    # count_g+=1
    return chat_completion.choices[0].message.content

def get_summaries_by_subcategory_id(subcategory_id):

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        query = """
        MATCH (s:SubAttribute {id: $subcategory_id})
        OPTIONAL MATCH (s)<-[:DESCRIBES]-(t:TextSummary)
        WITH s, collect(t) as summaries
        RETURN s.id as subcategory_id, 
               s.name as subcategory_name, 
               s.priority as priority,
               summaries
        """

        result = session.run(query, subcategory_id=subcategory_id)
        record = result.single()

        if not record:
            return None
        word_count=0
        summary_list =""
        for summary in record["summaries"]:
            if summary is not None:
                # print(summary)
                # exit(1)
                # summary_dict = dict(summary)
                summary_list+= summary["content"]
                word_count+=len(summary["content"].split())
        print("word count: "+str(word_count))
        return  [word_count, summary_list]

def groq_summary(key1,key2,text, llm, txt_length, count):
    if text=="":
        return ""
    # Construct the question string
    your_question = (
        f"Analyse the given text, Extract precise and concise information relevant to the subcategory '{key2}' under the category '{key1}' only. "
        "Focus only on meaningful and relevant details to the point.\n"
        "Here is the text:\n"
    )
    full_prompt = f"{your_question}{text}"
    response=""
    # print(full_prompt)
    print("text length: " + str(txt_length))
    if txt_length<=TOKEN_LIMIT:
        response = query_groq(full_prompt, groq_model_l, count)
        count+=1
    else:
        # Define the prompt template
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Can you summarize the following text concisely without missing any key points? {text}"
        )
        # Execute the chain
        chain = prompt | llm
        # response = chain.invoke({"text": text})
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }

        # Headers
        headers = {
            "Content-Type": "application/json"
        }

        req_response = requests.post(GEMINI_URL+GEMINI_API_KEY[count%g_l], headers=headers, data=json.dumps(payload))
        response = req_response.json()["candidates"][0]["content"]["parts"][0]["text"]
        count+=1
    print(response)
    print("\n\n")


    return response.replace("\n","")
for k1, v1 in data_attributes.items():
    for k2, v2 in v1.items():
        summaries=get_summaries_by_subcategory_id(v2)
        data_final=groq_summary(k1,k2,summaries[1], g_llm, summaries[0], count)
        v1[k2]=data_final

with open("data_final.json", "w") as f:
    json.dump(data_attributes, f, indent=2)

print("completed")

