# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re
# import spacy
# from neo4j import GraphDatabase
# import argparse

# def parse_medicare_chapter(pdf_path, chapter_num):
#     """Parse a Medicare document chapter with improved pattern matching"""
#     # Load the PDF
#     loader = PyPDFLoader(pdf_path)
#     pages = loader.load_and_split()
    
#     # Extract the specific chapter - MORE FLEXIBLE PATTERN
#     chapter_pattern = re.compile(f"Chapter\\s*{chapter_num}[:\\s]", re.IGNORECASE)
#     chapter_text = ""
#     chapter_found = False
    
#     for page in pages:
#         if chapter_pattern.search(page.page_content):
#             chapter_found = True
#             chapter_text += page.page_content
#         elif chapter_found and f"Chapter {chapter_num + 1}" not in page.page_content:
#             # Continue adding content until we hit the next chapter
#             chapter_text += page.page_content
    
#     # Parse sections
#     sections = []
#     section_pattern = re.compile(r"Section (\d+\.\d+) (.*)")
#     current_section = None
    
#     for line in chapter_text.split('\n'):
#         section_match = section_pattern.match(line)
#         if section_match:
#             if current_section:
#                 sections.append(current_section)
#             section_num, section_title = section_match.groups()
#             current_section = {
#                 "section_number": section_num,
#                 "title": section_title,
#                 "content": ""
#             }
#         elif current_section:
#             current_section["content"] += line + "\n"
    
#     if current_section:
#         sections.append(current_section)
    
#     # Create chapter structure
#     chapter = {
#         "chapter_number": chapter_num,
#         "title": f"Chapter {chapter_num}",
#         "sections": sections
#     }
    
#     return chapter

# def extract_entities(chapter_data):
#     """Extract key entities from chapter content"""
#     try:
#         nlp = spacy.load("en_core_web_lg")
#     except OSError:
#         # Fallback to smaller model if large model isn't available
#         try:
#             nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("Please install spaCy models with: python -m spacy download en_core_web_sm")
#             return {"requirements": [], "service_areas": [], "payment_options": [], "contacts": []}
    
#     entities = {
#         "requirements": [],
#         "service_areas": [],
#         "payment_options": [],
#         "contacts": []
#     }
    
#     # Process each section for entities
#     for section in chapter_data["sections"]:
#         doc = nlp(section["content"])
        
#         # Extract key requirements
#         if "eligibility" in section["title"].lower() or "requirement" in section["title"].lower():
#             requirements = [sent.text for sent in doc.sents if "required" in sent.text or "eligible" in sent.text]
#             entities["requirements"].extend(requirements)
        
#         # Extract service areas
#         if "service area" in section["title"].lower() or "geographic" in section["title"].lower():
#             for ent in doc.ents:
#                 if ent.label_ == "GPE" or ent.label_ == "LOC":
#                     entities["service_areas"].append(ent.text)
        
#         # Extract payment options
#         if "premium" in section["title"].lower() or "payment" in section["title"].lower() or "pay" in section["title"].lower():
#             payment_sentences = [sent.text for sent in doc.sents if "pay" in sent.text or "payment" in sent.text]
#             entities["payment_options"].extend(payment_sentences)
        
#         # Extract contact information
#         phone_pattern = re.compile(r"\d{1}-\d{3}-\d{3}-\d{4}")
#         for sent in doc.sents:
#             if phone_pattern.search(sent.text) or "call" in sent.text.lower():
#                 entities["contacts"].append(sent.text)
    
#     return entities

# def populate_neo4j(uri, username, password, chapter_data, entities):
#     """Populate Neo4j database with chapter data and entities"""
#     driver = GraphDatabase.driver(uri, auth=(username, password))
    
#     with driver.session() as session:
#         # Create Document node if not exists
#         session.run("""
#             MERGE (d:Document {title: $title})
#         """, title="AARP Medicare Advantage Evidence of Coverage")
        
#         # Create Chapter node
#         session.run("""
#             MATCH (d:Document {title: $doc_title})
#             MERGE (c:Chapter {
#                 number: $chapter_num,
#                 title: $chapter_title
#             })
#             MERGE (d)-[:CONTAINS]->(c)
#         """, doc_title="AARP Medicare Advantage Evidence of Coverage",
#              chapter_num=chapter_data["chapter_number"],
#              chapter_title=chapter_data["title"])
        
#         # Create Section nodes
#         for section in chapter_data["sections"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num})
#                 MERGE (s:Section {
#                     number: $section_num,
#                     title: $section_title
#                 })
#                 MERGE (c)-[:CONTAINS]->(s)
#                 SET s.content = $content
#             """, chapter_num=chapter_data["chapter_number"],
#                  section_num=section["section_number"],
#                  section_title=section["title"],
#                  content=section["content"])
        
#         # Create entities
#         for req in entities["requirements"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num})
#                 MERGE (r:Requirement {text: $req_text})
#                 MERGE (c)-[:CONTAINS]->(r)
#             """, chapter_num=chapter_data["chapter_number"], req_text=req)
        
#         for area in entities["service_areas"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num})
#                 MERGE (a:ServiceArea {name: $area_name})
#                 MERGE (c)-[:COVERS]->(a)
#             """, chapter_num=chapter_data["chapter_number"], area_name=area)
        
#         # Add cross-references between sections
#         for section in chapter_data["sections"]:
#             content = section["content"]
#             # Find references to other sections
#             ref_pattern = re.compile(r"Chapter (\d+), Section (\d+\.\d+)")
#             for match in ref_pattern.finditer(content):
#                 ref_chapter, ref_section = match.groups()
#                 session.run("""
#                     MATCH (s1:Section {number: $section_num})
#                     MATCH (s2:Section {number: $ref_section})
#                     MERGE (s1)-[:REFERENCES]->(s2)
#                 """, section_num=section["section_number"], 
#                      ref_section=ref_section)
    
#     driver.close()

# def generate_visualization_query():
#     """Generate Neo4j query for visualizing document structure"""
#     return """
#     MATCH path = (d:Document)-[:CONTAINS*1..3]->()
#     RETURN path LIMIT 100
#     """

# def process_medicare_document(pdf_path, neo4j_uri, neo4j_user, neo4j_password):
#     """Process a Medicare document and store in Neo4j"""
#     # Process each chapter - INCREASED RANGE
#     for chapter_num in range(1, 21):  # Support up to 20 chapters
#         print(f"Searching for Chapter {chapter_num}...")
        
#         # Parse chapter
#         chapter_data = parse_medicare_chapter(pdf_path, chapter_num)
        
#         if not chapter_data["sections"]:
#             print(f"No sections found for Chapter {chapter_num}")
#             continue  # Skip if chapter not found
        
#         print(f"Found {len(chapter_data['sections'])} sections in Chapter {chapter_num}")
        
#         # Extract entities
#         entities = extract_entities(chapter_data)
        
#         # Populate Neo4j
#         populate_neo4j(neo4j_uri, neo4j_user, neo4j_password, chapter_data, entities)
        
#         print(f"Processed Chapter {chapter_num}")
    
#     print("Document processing complete!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process Medicare document chapters into Neo4j')
#     parser.add_argument('--pdf', required=True, help='Path to the PDF file')
#     parser.add_argument('--uri', required=True, help='Neo4j connection URI')
#     parser.add_argument('--user', required=True, help='Neo4j username')
#     parser.add_argument('--password', required=True, help='Neo4j password')
    
#     args = parser.parse_args()
    
#     process_medicare_document(args.pdf, args.uri, args.user, args.password)






# #####below code is working for single pdf with qa using tinyllama_medicare_qa.py
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re
# import spacy
# from neo4j import GraphDatabase
# import argparse

# def parse_medicare_chapter(pdf_path, chapter_num):
#     """Parse a Medicare document chapter with improved pattern matching"""
#     # Load the PDF
#     loader = PyPDFLoader(pdf_path)
#     pages = loader.load_and_split()
    
#     # Extract the specific chapter - MORE FLEXIBLE PATTERN
#     chapter_pattern = re.compile(f"Chapter\\s*{chapter_num}[:\\s]", re.IGNORECASE)
#     chapter_text = ""
#     chapter_found = False
    
#     for page in pages:
#         if chapter_pattern.search(page.page_content):
#             chapter_found = True
#             chapter_text += page.page_content
#         elif chapter_found and f"Chapter {chapter_num + 1}" not in page.page_content:
#             # Continue adding content until we hit the next chapter
#             chapter_text += page.page_content
    
#     # Parse sections
#     sections = []
#     section_pattern = re.compile(r"Section (\d+\.\d+) (.*)")
#     current_section = None
    
#     for line in chapter_text.split('\n'):
#         section_match = section_pattern.match(line)
#         if section_match:
#             if current_section:
#                 sections.append(current_section)
#             section_num, section_title = section_match.groups()
#             current_section = {
#                 "section_number": section_num,
#                 "title": section_title,
#                 "content": ""
#             }
#         elif current_section:
#             current_section["content"] += line + "\n"
    
#     if current_section:
#         sections.append(current_section)
    
#     # Create chapter structure
#     chapter = {
#         "chapter_number": chapter_num,
#         "title": f"Chapter {chapter_num}",
#         "sections": sections
#     }
    
#     return chapter

# def extract_entities(chapter_data):
#     """Extract key entities from chapter content"""
#     try:
#         nlp = spacy.load("en_core_web_lg")
#     except OSError:
#         # Fallback to smaller model if large model isn't available
#         try:
#             nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("Please install spaCy models with: python -m spacy download en_core_web_sm")
#             return {"requirements": [], "service_areas": [], "payment_options": [], "contacts": []}
    
#     entities = {
#         "requirements": [],
#         "service_areas": [],
#         "payment_options": [],
#         "contacts": []
#     }
    
#     # Process each section for entities
#     for section in chapter_data["sections"]:
#         doc = nlp(section["content"])
        
#         # Extract key requirements
#         if "eligibility" in section["title"].lower() or "requirement" in section["title"].lower():
#             requirements = [sent.text for sent in doc.sents if "required" in sent.text or "eligible" in sent.text]
#             entities["requirements"].extend(requirements)
        
#         # Extract service areas
#         if "service area" in section["title"].lower() or "geographic" in section["title"].lower():
#             for ent in doc.ents:
#                 if ent.label_ == "GPE" or ent.label_ == "LOC":
#                     entities["service_areas"].append(ent.text)
        
#         # Extract payment options
#         if "premium" in section["title"].lower() or "payment" in section["title"].lower() or "pay" in section["title"].lower():
#             payment_sentences = [sent.text for sent in doc.sents if "pay" in sent.text or "payment" in sent.text]
#             entities["payment_options"].extend(payment_sentences)
        
#         # Extract contact information
#         phone_pattern = re.compile(r"\d{1}-\d{3}-\d{3}-\d{4}")
#         for sent in doc.sents:
#             if phone_pattern.search(sent.text) or "call" in sent.text.lower():
#                 entities["contacts"].append(sent.text)
    
#     return entities

# def populate_neo4j(uri, username, password, chapter_data, entities):
#     """Populate Neo4j database with chapter data and entities"""
#     driver = GraphDatabase.driver(uri, auth=(username, password))
    
#     with driver.session() as session:
#         # Create Document node if not exists
#         session.run("""
#             MERGE (d:Document {title: $title})
#         """, title="AARP Medicare Advantage Evidence of Coverage")
        
#         # Create Chapter node
#         session.run("""
#             MATCH (d:Document {title: $doc_title})
#             MERGE (c:Chapter {
#                 number: $chapter_num,
#                 title: $chapter_title
#             })
#             MERGE (d)-[:CONTAINS]->(c)
#         """, doc_title="AARP Medicare Advantage Evidence of Coverage",
#              chapter_num=chapter_data["chapter_number"],
#              chapter_title=chapter_data["title"])
        
#         # Create Section nodes
#         for section in chapter_data["sections"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num})
#                 MERGE (s:Section {
#                     number: $section_num,
#                     title: $section_title
#                 })
#                 MERGE (c)-[:CONTAINS]->(s)
#                 SET s.content = $content
#             """, chapter_num=chapter_data["chapter_number"],
#                  section_num=section["section_number"],
#                  section_title=section["title"],
#                  content=section["content"])
        
#         # Create entities
#         for req in entities["requirements"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num})
#                 MERGE (r:Requirement {text: $req_text})
#                 MERGE (c)-[:CONTAINS]->(r)
#             """, chapter_num=chapter_data["chapter_number"], req_text=req)
        
#         for area in entities["service_areas"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num})
#                 MERGE (a:ServiceArea {name: $area_name})
#                 MERGE (c)-[:COVERS]->(a)
#             """, chapter_num=chapter_data["chapter_number"], area_name=area)
        
#         # Add cross-references between sections
#         for section in chapter_data["sections"]:
#             content = section["content"]
#             # Find references to other sections
#             ref_pattern = re.compile(r"Chapter (\d+), Section (\d+\.\d+)")
#             for match in ref_pattern.finditer(content):
#                 ref_chapter, ref_section = match.groups()
#                 session.run("""
#                     MATCH (s1:Section {number: $section_num})
#                     MATCH (s2:Section {number: $ref_section})
#                     MERGE (s1)-[:REFERENCES]->(s2)
#                 """, section_num=section["section_number"], 
#                      ref_section=ref_section)
    
#     driver.close()

# def generate_visualization_query():
#     """Generate Neo4j query for visualizing document structure"""
#     return """
#     MATCH path = (d:Document)-[:CONTAINS*1..3]->()
#     RETURN path LIMIT 100
#     """

# def process_medicare_document(pdf_path, neo4j_uri, neo4j_user, neo4j_password):
#     """Process a Medicare document and store in Neo4j"""
#     # Process each chapter - INCREASED RANGE
#     for chapter_num in range(1, 21):  # Support up to 20 chapters
#         print(f"Searching for Chapter {chapter_num}...")
        
#         # Parse chapter
#         chapter_data = parse_medicare_chapter(pdf_path, chapter_num)
        
#         if not chapter_data["sections"]:
#             print(f"No sections found for Chapter {chapter_num}")
#             continue  # Skip if chapter not found
        
#         print(f"Found {len(chapter_data['sections'])} sections in Chapter {chapter_num}")
        
#         # Extract entities
#         entities = extract_entities(chapter_data)
        
#         # Populate Neo4j
#         populate_neo4j(neo4j_uri, neo4j_user, neo4j_password, chapter_data, entities)
        
#         print(f"Processed Chapter {chapter_num}")
    
#     print("Document processing complete!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process Medicare document chapters into Neo4j')
#     parser.add_argument('--pdf', required=True, help='Path to the PDF file')
#     parser.add_argument('--uri', required=True, help='Neo4j connection URI')
#     parser.add_argument('--user', required=True, help='Neo4j username')
#     parser.add_argument('--password', required=True, help='Neo4j password')
    
#     args = parser.parse_args()
    
#     process_medicare_document(args.pdf, args.uri, args.user, args.password)





# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re
# import spacy
# from neo4j import GraphDatabase
# import argparse

# def parse_medicare_chapter(pdf_path, chapter_num):
#     """Parse a Medicare document chapter with improved pattern matching"""
#     # Load the PDF
#     loader = PyPDFLoader(pdf_path)
#     pages = loader.load_and_split()
    
#     # Extract the specific chapter - MORE FLEXIBLE PATTERN
#     chapter_pattern = re.compile(f"Chapter\\s*{chapter_num}[:\\s]", re.IGNORECASE)
#     chapter_text = ""
#     chapter_found = False
    
#     for page in pages:
#         if chapter_pattern.search(page.page_content):
#             chapter_found = True
#             chapter_text += page.page_content
#         elif chapter_found and f"Chapter {chapter_num + 1}" not in page.page_content:
#             # Continue adding content until we hit the next chapter
#             chapter_text += page.page_content
    
#     # Parse sections
#     sections = []
#     section_pattern = re.compile(r"Section (\d+\.\d+) (.*)")
#     current_section = None
    
#     for line in chapter_text.split('\n'):
#         section_match = section_pattern.match(line)
#         if section_match:
#             if current_section:
#                 sections.append(current_section)
#             section_num, section_title = section_match.groups()
#             current_section = {
#                 "section_number": section_num,
#                 "title": section_title,
#                 "content": ""
#             }
#         elif current_section:
#             current_section["content"] += line + "\n"
    
#     if current_section:
#         sections.append(current_section)
    
#     # Create chapter structure
#     chapter = {
#         "chapter_number": chapter_num,
#         "title": f"Chapter {chapter_num}",
#         "sections": sections
#     }
    
#     return chapter

# def extract_entities(chapter_data):
#     """Extract key entities from chapter content"""
#     try:
#         nlp = spacy.load("en_core_web_lg")
#     except OSError:
#         # Fallback to smaller model if large model isn't available
#         try:
#             nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("Please install spaCy models with: python -m spacy download en_core_web_sm")
#             return {"requirements": [], "service_areas": [], "payment_options": [], "contacts": []}
    
#     entities = {
#         "requirements": [],
#         "service_areas": [],
#         "payment_options": [],
#         "contacts": []
#     }
    
#     # Process each section for entities
#     for section in chapter_data["sections"]:
#         doc = nlp(section["content"])
        
#         # Extract key requirements
#         if "eligibility" in section["title"].lower() or "requirement" in section["title"].lower():
#             requirements = [sent.text for sent in doc.sents if "required" in sent.text or "eligible" in sent.text]
#             entities["requirements"].extend(requirements)
        
#         # Extract service areas
#         if "service area" in section["title"].lower() or "geographic" in section["title"].lower():
#             for ent in doc.ents:
#                 if ent.label_ == "GPE" or ent.label_ == "LOC":
#                     entities["service_areas"].append(ent.text)
        
#         # Extract payment options
#         if "premium" in section["title"].lower() or "payment" in section["title"].lower() or "pay" in section["title"].lower():
#             payment_sentences = [sent.text for sent in doc.sents if "pay" in sent.text or "payment" in sent.text]
#             entities["payment_options"].extend(payment_sentences)
        
#         # Extract contact information
#         phone_pattern = re.compile(r"\d{1}-\d{3}-\d{3}-\d{4}")
#         for sent in doc.sents:
#             if phone_pattern.search(sent.text) or "call" in sent.text.lower():
#                 entities["contacts"].append(sent.text)
    
#     return entities

# def populate_neo4j(uri, username, password, chapter_data, entities, doc_title):
#     """Populate Neo4j database with chapter data and entities"""
#     driver = GraphDatabase.driver(uri, auth=(username, password))
    
#     with driver.session() as session:
#         # Create Document node if not exists
#         session.run("""
#             MERGE (d:Document {title: $title})
#         """, title=doc_title)
        
#         # Create Chapter node
#         session.run("""
#             MATCH (d:Document {title: $doc_title})
#             MERGE (c:Chapter {
#                 number: $chapter_num,
#                 title: $chapter_title,
#                 document: $doc_title
#             })
#             MERGE (d)-[:CONTAINS]->(c)
#         """, doc_title=doc_title,
#              chapter_num=chapter_data["chapter_number"],
#              chapter_title=chapter_data["title"])
        
#         # Create Section nodes
#         for section in chapter_data["sections"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num, document: $doc_title})
#                 MERGE (s:Section {
#                     number: $section_num,
#                     title: $section_title,
#                     document: $doc_title
#                 })
#                 MERGE (c)-[:CONTAINS]->(s)
#                 SET s.content = $content
#             """, chapter_num=chapter_data["chapter_number"],
#                  section_num=section["section_number"],
#                  section_title=section["title"],
#                  content=section["content"],
#                  doc_title=doc_title)
        
#         # Create entities
#         for req in entities["requirements"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num, document: $doc_title})
#                 MERGE (r:Requirement {text: $req_text, document: $doc_title})
#                 MERGE (c)-[:CONTAINS]->(r)
#             """, chapter_num=chapter_data["chapter_number"], req_text=req, doc_title=doc_title)
        
#         for area in entities["service_areas"]:
#             session.run("""
#                 MATCH (c:Chapter {number: $chapter_num, document: $doc_title})
#                 MERGE (a:ServiceArea {name: $area_name, document: $doc_title})
#                 MERGE (c)-[:COVERS]->(a)
#             """, chapter_num=chapter_data["chapter_number"], area_name=area, doc_title=doc_title)
        
#         # Add cross-references between sections
#         for section in chapter_data["sections"]:
#             content = section["content"]
#             # Find references to other sections
#             ref_pattern = re.compile(r"Chapter (\d+), Section (\d+\.\d+)")
#             for match in ref_pattern.finditer(content):
#                 ref_chapter, ref_section = match.groups()
#                 session.run("""
#                     MATCH (s1:Section {number: $section_num, document: $doc_title})
#                     MATCH (s2:Section {number: $ref_section, document: $doc_title})
#                     MERGE (s1)-[:REFERENCES]->(s2)
#                 """, section_num=section["section_number"], 
#                      ref_section=ref_section,
#                      doc_title=doc_title)
    
#     driver.close()

# def generate_visualization_query(doc_title):
#     """Generate Neo4j query for visualizing document structure"""
#     return f"""
#     MATCH path = (d:Document {{title: "{doc_title}"}})-[:CONTAINS*1..3]->()
#     RETURN path LIMIT 100
#     """

# def process_medicare_document(pdf_path, neo4j_uri, neo4j_user, neo4j_password, doc_title):
#     """Process a Medicare document and store in Neo4j"""
#     print(f"Processing document: {doc_title}")
    
#     # Process each chapter - INCREASED RANGE
#     for chapter_num in range(1, 21):  # Support up to 20 chapters
#         print(f"Searching for Chapter {chapter_num}...")
        
#         # Parse chapter
#         chapter_data = parse_medicare_chapter(pdf_path, chapter_num)
        
#         if not chapter_data["sections"]:
#             print(f"No sections found for Chapter {chapter_num}")
#             continue  # Skip if chapter not found
        
#         print(f"Found {len(chapter_data['sections'])} sections in Chapter {chapter_num}")
        
#         # Extract entities
#         entities = extract_entities(chapter_data)
        
#         # Populate Neo4j
#         populate_neo4j(neo4j_uri, neo4j_user, neo4j_password, chapter_data, entities, doc_title)
        
#         print(f"Processed Chapter {chapter_num}")
    
#     print(f"Document processing complete for: {doc_title}")
#     print(f"To visualize this document, run this query in Neo4j Browser:")
#     print(generate_visualization_query(doc_title))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process Medicare document chapters into Neo4j')
#     parser.add_argument('--pdf', required=True, help='Path to the PDF file')
#     parser.add_argument('--uri', required=True, help='Neo4j connection URI')
#     parser.add_argument('--user', required=True, help='Neo4j username')
#     parser.add_argument('--password', required=True, help='Neo4j password')
#     parser.add_argument('--doc-title', required=True, help='Title of the document')
    
#     args = parser.parse_args()
    
#     process_medicare_document(args.pdf, args.uri, args.user, args.password, args.doc_title)



from langchain_community.document_loaders import PyPDFLoader
import re
import spacy
from neo4j import GraphDatabase
import argparse
import unicodedata

def normalize_text(text):
    """Normalize text to improve pattern matching"""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove non-alphanumeric characters except spaces, periods, commas, and dollar signs
    text = re.sub(r'[^\w\s.,\$]', ' ', text)
    # Trim whitespace
    text = text.strip()
    return text

def get_surrounding_text(text, keyword, radius=50):
    """Get text surrounding a keyword"""
    start_idx = text.find(keyword)
    if start_idx == -1:
        return ""
    
    start = max(0, start_idx - radius)
    end = min(len(text), start_idx + len(keyword) + radius)
    return text[start:end]

def extract_specific_features(chapter_data):
    """Dedicated function to extract specific features from all sections"""
    # Initialize all features as "Not Found"
    features = {
        "vision": "Not Found",
        "dental": "Not Found",
        "hearing": "Not Found",
        "transportation": "Not Found",
        "fitness": "Not Found",
        "worldwide_emergency": "Not Found", 
        "otc": "Not Found",  # Over the counter
        "in_home_support": "Not Found",
        "emergency_response": "Not Found",
        "bathroom_safety": "Not Found",
        "meals": "Not Found",
        "physical_exams": "Not Found",
        "telehealth": "Not Found",
        "endodontics": "Not Found",
        "periodontics": "Not Found"
    }
    
    # Specific keywords to search for
    feature_keywords = {
        "vision": ["vision", "eye exam", "eyewear", "glasses", "eye care"],
        "dental": ["dental", "teeth", "oral health", "tooth", "dentist"],
        "hearing": ["hearing", "ear exam", "hearing aid", "audiology"],
        "transportation": ["transportation", "rides", "transit", "ride to appointment"],
        "fitness": ["fitness", "gym", "exercise", "silver sneakers", "wellness program"],
        "worldwide_emergency": ["worldwide", "international", "travel", "foreign", "outside us"],
        "otc": ["over the counter", "otc", "non-prescription", "health products"],
        "in_home_support": ["home support", "in-home", "home health aide", "caregiver"],
        "emergency_response": ["emergency response", "medical alert", "alert system", "emergency button"],
        "bathroom_safety": ["bathroom safety", "grab bars", "shower seat", "safety equipment"],
        "meals": ["meal", "food delivery", "nutritional support", "home delivered meals"],
        "physical_exams": ["physical exam", "annual visit", "wellness exam", "routine physical"],
        "telehealth": ["telehealth", "virtual", "online visit", "video appointment", "remote care"],
        "endodontics": ["endodontics", "root canal", "dental procedure", "dental surgery"],
        "periodontics": ["periodontics", "gum", "gum disease", "deep cleaning", "scaling"]
    }
    
    # Process each section for feature mentions
    for section in chapter_data["sections"]:
        content = section["content"].lower()
        title = section["title"].lower() if section["title"] else ""
        combined_text = title + " " + content
        
        # Look for each feature
        for feature, keywords in feature_keywords.items():
            if features[feature] != "Available" and features[feature] != "Not Available":
                for keyword in keywords:
                    if keyword in combined_text:
                        # Check if it's described as not available
                        not_patterns = [
                            f"not {keyword}",
                            f"{keyword} not",
                            f"{keyword} is not",
                            f"no {keyword}",
                            "not available",
                            "not covered",
                            "not included",
                            "excluded"
                        ]
                        
                        if any(not_pat in combined_text for not_pat in not_patterns):
                            features[feature] = "Not Available"
                            break
                        else:
                            features[feature] = "Available"
                            break
        
        # Special case for summary tables/sections - these often have all benefits listed
        benefits_section_keywords = ["summary of benefits", "benefits at a glance", "covered benefits"]
        if any(keyword in combined_text for keyword in benefits_section_keywords):
            # This is likely a summary section - scan it for all features
            for feature, keywords in feature_keywords.items():
                if features[feature] != "Available" and features[feature] != "Not Available":
                    for keyword in keywords:
                        if keyword in combined_text:
                            proximity_text = get_surrounding_text(combined_text, keyword, 50)
                            
                            # Check if nearby text indicates it's not available
                            if "not" in proximity_text or "no" in proximity_text or "excluded" in proximity_text:
                                features[feature] = "Not Available"
                                break
                            else:
                                features[feature] = "Available"
                                break

    # Convert string values to boolean for Neo4j
    boolean_features = {}
    for feature, status in features.items():
        if status == "Available":
            boolean_features[feature] = True
        elif status == "Not Available":
            boolean_features[feature] = False
        # Keep "Not Found" as None
    
    return boolean_features

def identify_plan_metadata(section_content, section_title):
    """Identify and extract key plan metadata from section content with improved pattern matching"""
    metadata = {}
    
    # Normalize content for better pattern matching
    normalized_content = normalize_text(section_content)
    normalized_title = normalize_text(section_title) if section_title else ""
    
    # Check for premium information
    premium_patterns = [
        r'(monthly|health|plan|medical) premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'premium.*?(is|of) \$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'pay.*?premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'premium payment of \$(\d+(?:,\d+)?(?:\.\d+)?)'
    ]
    
    for pattern in premium_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            if isinstance(matches[0], tuple):
                metadata["premium"] = matches[0][1]  # Get the second capture group
            else:
                metadata["premium"] = matches[0]
            break
    
    # Check for drug premium information
    drug_premium_patterns = [
        r'(drug|part d|prescription) premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'premium for (drug|part d|prescription).*?\$(\d+(?:,\d+)?(?:\.\d+)?)'
    ]
    
    for pattern in drug_premium_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            if isinstance(matches[0], tuple):
                metadata["drug_premium"] = matches[0][1]  # Get the second capture group
            else:
                metadata["drug_premium"] = matches[0]
            break
    
    # Check for part B premium information
    part_b_patterns = [
        r'part b premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'medicare part b premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)'
    ]
    
    for pattern in part_b_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            metadata["part_b_premium"] = matches[0]
            break
    
    # Check for deductible information
    deductible_patterns = [
        r'(health|medical|plan) deductible.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'deductible (is|of) \$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'deductible.*?\$(\d+(?:,\d+)?(?:\.\d+)?)'
    ]
    
    for pattern in deductible_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            if isinstance(matches[0], tuple):
                metadata["deductible"] = matches[0][1]  # Get the second capture group
            else:
                metadata["deductible"] = matches[0]
            break
    
    # Check for drug deductible
    drug_deductible_patterns = [
        r'(drug|part d|prescription) deductible.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'deductible for (drug|part d|prescription).*?\$(\d+(?:,\d+)?(?:\.\d+)?)'
    ]
    
    for pattern in drug_deductible_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            if isinstance(matches[0], tuple):
                metadata["drug_deductible"] = matches[0][1]  # Get the second capture group
            else:
                metadata["drug_deductible"] = matches[0]
            break
    
    # Check for maximum out-of-pocket
    moop_patterns = [
        r'maximum out.?of.?pocket.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'out.?of.?pocket (maximum|limit).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'moop.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
        r'maximum amount you (pay|will pay).*?\$(\d+(?:,\d+)?(?:\.\d+)?)'
    ]
    
    for pattern in moop_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            if isinstance(matches[0], tuple):
                # Check if the second item exists (might be a tuple from capture groups)
                if len(matches[0]) > 1:
                    metadata["max_out_of_pocket"] = matches[0][1]
                else:
                    metadata["max_out_of_pocket"] = matches[0][0]
            else:
                metadata["max_out_of_pocket"] = matches[0]
            break
    
    # Check for star rating
    star_patterns = [
        r'(\d+(?:\.\d+)?)\s*stars?',
        r'star rating.*?(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?).?star rating',
        r'rated (\d+(?:\.\d+)?) stars?'
    ]
    
    for pattern in star_patterns:
        matches = re.findall(pattern, normalized_content)
        if matches:
            metadata["star_rating"] = matches[0]
            break
    
    return metadata

def extract_benefit_costs(section_content):
    """Extract benefit costs from section content with improved pattern matching"""
    benefit_costs = {}
    
    # Normalize content
    normalized_content = normalize_text(section_content)
    
    benefit_patterns = {
        "primary_care": [
            r'primary (care|doctor).*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'primary (care|doctor).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'pcp visit.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'primary (care|doctor).*?covered'
        ],
        "specialist": [
            r'specialist.*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'specialist visit.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'specialist.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'specialist.*?covered'
        ],
        "emergency": [
            r'emergency (care|room|services).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'emergency (care|room|services).*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'er visit.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'emergency (care|room|services).*?covered'
        ],
        "inpatient": [
            r'inpatient (hospital|care).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'hospital stay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'inpatient (hospital|care).*?covered'
        ],
        "outpatient": [
            r'outpatient (hospital|services|surgery).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'outpatient (hospital|services|surgery).*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'outpatient (hospital|services|surgery).*?covered'
        ],
        "lab": [
            r'lab (services|tests).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'laboratory.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'lab (services|tests).*?covered'
        ],
        "xray": [
            r'x.?ray.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'radiograph.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'x.?ray.*?covered'
        ],
        "diagnostic": [
            r'diagnostic (tests|procedures).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'diagnostic (tests|procedures).*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'diagnostic (tests|procedures).*?covered'
        ],
        "urgent_care": [
            r'urgent care.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'urgent care.*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'urgent care.*?covered'
        ],
        "preventive": [
            r'preventive (services|care).*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'preventive (services|care).*?copay.*?\$(\d+(?:,\d+)?(?:\.\d+)?)',
            r'preventive (services|care).*?covered'
        ]
    }
    
    for benefit, patterns in benefit_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, normalized_content)
            if matches:
                if isinstance(matches[0], tuple):
                    # If it's a tuple, get the dollar amount (second capture group)
                    if len(matches[0]) > 1:
                        benefit_costs[benefit] = matches[0][1]
                    else:
                        # It's a "covered" pattern without a dollar amount
                        benefit_costs[benefit] = "covered"
                else:
                    # If not a tuple, it's from a "covered" pattern
                    benefit_costs[benefit] = "covered"
                break
        
        # Check for $0 or free benefits
        if benefit not in benefit_costs:
            zero_patterns = [
                rf'{benefit.replace("_", " ")}.*?(\$0|no copay|0 copay|free|fully covered)',
                rf'{benefit.replace("_", " ")}.*?covered at 100%'
            ]
            for pattern in zero_patterns:
                if re.search(pattern, normalized_content):
                    benefit_costs[benefit] = "0"
                    break
    
    return benefit_costs

def identify_available_features(section_content):
    """Identify available features from section content with improved pattern matching"""
    features = {}
    
    # Normalize content
    normalized_content = normalize_text(section_content)
    
    feature_patterns = {
        "vision": [
            r'vision (benefit|coverage|services)',
            r'eye exam',
            r'eyewear',
            r'glasses',
            r'vision care'
        ],
        "dental": [
            r'dental (benefit|coverage|services|care)',
            r'oral health',
            r'teeth cleaning',
            r'dental exam'
        ],
        "hearing": [
            r'hearing (benefit|coverage|services|aid|exam)',
            r'hearing care',
            r'hearing test'
        ],
        "transportation": [
            r'transportation (benefit|service)',
            r'rides to',
            r'medical transport',
            r'non.?emergency (transportation|transport)'
        ],
        "fitness": [
            r'fitness (benefit|program)',
            r'gym membership',
            r'silver ?sneakers',
            r'exercise program'
        ],
        "worldwide_emergency": [
            r'worldwide emergency',
            r'emergency (outside|abroad|international)',
            r'international emergency',
            r'global emergency coverage'
        ],
        "otc": [
            r'over.?the.?counter',
            r'otc (benefit|allowance|items|products)',
            r'otc drugs'
        ],
        "telehealth": [
            r'telehealth',
            r'virtual (visit|care|consultation)',
            r'remote access',
            r'telemedicine'
        ],
        "meals": [
            r'meal (benefit|service|delivery)',
            r'post.?discharge meal',
            r'meal support',
            r'nutritional (support|meals)'
        ],
        "physical_exams": [
            r'physical exam',
            r'annual exam',
            r'wellness visit',
            r'routine physical'
        ],
        "in_home_support": [
            r'in.?home support',
            r'home health aide',
            r'caregiver support',
            r'home.?based support'
        ],
        "emergency_response": [
            r'emergency response',
            r'medical alert',
            r'personal emergency',
            r'pers device'
        ],
        "bathroom_safety": [
            r'bathroom safety',
            r'safety device',
            r'grab bars',
            r'shower seat',
            r'bathroom modification'
        ],
        "endodontics": [
            r'endodontic',
            r'root canal',
            r'dental procedure',
            r'tooth pulp'
        ],
        "periodontics": [
            r'periodontic',
            r'gum disease',
            r'gum treatment',
            r'periodontal'
        ]
    }
    
    for feature, patterns in feature_patterns.items():
        feature_found = False
        for pattern in patterns:
            if re.search(pattern, normalized_content):
                feature_found = True
                break
        
        if feature_found:
            # Check if feature is specifically listed as not available
            not_available_patterns = [
                rf'{pattern}.*?not (available|covered|included)',
                rf'does not (cover|include).*?{pattern}',
                rf'no {pattern}',
                rf'{pattern}.*?excluded'
            ]
            
            is_not_available = False
            for not_pattern in not_available_patterns:
                if re.search(not_pattern, normalized_content):
                    is_not_available = True
                    break
            
            features[feature] = not is_not_available
    
    return features

def parse_medicare_chapter(pdf_path, chapter_num):
    """Parse a Medicare document chapter with improved pattern matching"""
    # Load the PDF
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
                "section_number": section_num,
                "title": section_title,
                "content": ""
            }
        elif current_section:
            current_section["content"] += line + "\n"
    
    if current_section:
        sections.append(current_section)
    
    # Create chapter structure
    chapter = {
        "chapter_number": chapter_num,
        "title": f"Chapter {chapter_num}",
        "sections": sections
    }
    
    return chapter

def extract_entities(chapter_data):
    """Extract key entities from chapter content"""
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        # Fallback to smaller model if large model isn't available
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy models with: python -m spacy download en_core_web_sm")
            return {"requirements": [], "service_areas": [], "payment_options": [], "contacts": []}
    
    entities = {
        "requirements": [],
        "service_areas": [],
        "payment_options": [],
        "contacts": []
    }
    
    # Process each section for entities
    for section in chapter_data["sections"]:
        doc = nlp(section["content"])
        
        # Extract key requirements
        if "eligibility" in section["title"].lower() or "requirement" in section["title"].lower():
            requirements = [sent.text for sent in doc.sents if "required" in sent.text or "eligible" in sent.text]
            entities["requirements"].extend(requirements)
        
        # Extract service areas
        if "service area" in section["title"].lower() or "geographic" in section["title"].lower():
            for ent in doc.ents:
                if ent.label_ == "GPE" or ent.label_ == "LOC":
                    entities["service_areas"].append(ent.text)
        
        # Extract payment options
        if "premium" in section["title"].lower() or "payment" in section["title"].lower() or "pay" in section["title"].lower():
            payment_sentences = [sent.text for sent in doc.sents if "pay" in sent.text or "payment" in sent.text]
            entities["payment_options"].extend(payment_sentences)
        
        # Extract contact information
        phone_pattern = re.compile(r"\d{1}-\d{3}-\d{3}-\d{4}")
        for sent in doc.sents:
            if phone_pattern.search(sent.text) or "call" in sent.text.lower():
                entities["contacts"].append(sent.text)
    
    return entities

def extract_features_from_summary_tables(pdf_path, doc_title, neo4j_uri, neo4j_user, neo4j_password):
    """Look specifically for summary tables that list features"""
    print("Searching for feature summary tables...")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Keywords that often appear in summary tables
    summary_keywords = [
        "summary of benefits", 
        "benefits and coverage", 
        "covered benefits",
        "additional benefits",
        "extra benefits",
        "optional benefits"
    ]
    
    # Feature keywords to look for
    feature_keywords = {
        "vision": ["vision", "eye", "eyewear"],
        "dental": ["dental", "teeth", "oral"],
        "hearing": ["hearing", "ear", "audiology"],
        "transportation": ["transportation", "rides", "transit"],
        "fitness": ["fitness", "gym", "exercise", "silver sneakers"],
        "worldwide_emergency": ["worldwide", "international", "travel", "foreign"],
        "otc": ["over the counter", "otc", "non-prescription"],
        "in_home_support": ["home support", "in-home", "aide"],
        "emergency_response": ["emergency response", "alert", "button"],
        "bathroom_safety": ["bathroom", "grab bars", "shower seat"],
        "meals": ["meal", "food", "nutrition"],
        "physical_exams": ["physical", "annual visit", "wellness exam"],
        "telehealth": ["telehealth", "virtual", "online", "video"],
        "endodontics": ["endodontics", "root canal"],
        "periodontics": ["periodontics", "gum", "scaling"]
    }
    
    found_features = {}
    
    # Search each page for summary tables
    for page in pages:
        content = page.page_content.lower()
        
        # Check if this page has summary table content
        if any(keyword in content for keyword in summary_keywords):
            print(f"Found potential summary table")
            
            # Check for each feature
            for feature, keywords in feature_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        # Check surrounding text for availability indicators
                        surrounding = get_surrounding_text(content, keyword, 100)
                        
                        if ("not" in surrounding and "available" in surrounding) or "excluded" in surrounding:
                            found_features[feature] = False  # Not Available
                            break
                        elif "available" in surrounding or "covered" in surrounding or "included" in surrounding:
                            found_features[feature] = True  # Available
                            break
                        else:
                            # Default to available if mentioned in summary and no negative indicators
                            found_features[feature] = True
    
    # Store found features in Neo4j
    if found_features:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            for feature, available in found_features.items():
                session.run("""
                    MATCH (d:Document {title: $doc_title})
                    MERGE (f:Feature {
                        name: $feature,
                        available: $available,
                        document: $doc_title
                    })
                    MERGE (d)-[:HAS_FEATURE]->(f)
                """, 
                     doc_title=doc_title,
                     feature=feature,
                     available=available)
        
        driver.close()
    
    return len(found_features) > 0

def populate_neo4j(uri, username, password, chapter_data, entities, doc_title):
    """Populate Neo4j database with chapter data, entities, and enhanced metadata"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Extract document-wide metadata from sections
    plan_metadata = {}
    for section in chapter_data["sections"]:
        metadata = identify_plan_metadata(section["content"], section["title"])
        plan_metadata.update(metadata)
    
    # Extract features across all sections
    document_features = extract_specific_features(chapter_data)
    
    with driver.session() as session:
        # Create fulltext search index if it doesn't exist
        try:
            session.run("""
                CALL db.index.fulltext.createNodeIndex(
                    'sectionContentIndex',
                    ['Section'],
                    ['content', 'title']
                )
            """)
            print("Created fulltext search index for Section nodes")
        except:
            # Index might already exist
            print("Fulltext search index might already exist")
        
        # Create Document node with metadata
        metadata_props = ""
        metadata_params = {"title": doc_title}
        
        if plan_metadata:
            metadata_props_list = []
            for key, value in plan_metadata.items():
                metadata_props_list.append(f"{key}: ${key}")
                metadata_params[key] = value
            
            if metadata_props_list:
                metadata_props = ", " + ", ".join(metadata_props_list)
        
        session.run(f"""
            MERGE (d:Document {{title: $title{metadata_props}}})
        """, **metadata_params)
        
        # Create Chapter node
        session.run("""
            MATCH (d:Document {title: $doc_title})
            MERGE (c:Chapter {
                number: $chapter_num,
                title: $chapter_title,
                document: $doc_title
            })
            MERGE (d)-[:CONTAINS]->(c)
        """, doc_title=doc_title,
             chapter_num=chapter_data["chapter_number"],
             chapter_title=chapter_data["title"])
        
        # Create Section nodes with enhanced metadata
        for section in chapter_data["sections"]:
            # Extract section-specific metadata
            section_metadata = identify_plan_metadata(section["content"], section["title"])
            metadata_props = ""
            metadata_params = {
                "chapter_num": chapter_data["chapter_number"],
                "section_num": section["section_number"],
                "section_title": section["title"],
                "content": section["content"],
                "doc_title": doc_title
            }
            
            if section_metadata:
                metadata_props_list = []
                for key, value in section_metadata.items():
                    metadata_props_list.append(f"{key}: ${key}")
                    metadata_params[key] = value
                
                if metadata_props_list:
                    metadata_props = ", " + ", ".join(metadata_props_list)
            
            # Create the section node with metadata
            session.run(f"""
                MATCH (c:Chapter {{number: $chapter_num, document: $doc_title}})
                MERGE (s:Section {{
                    number: $section_num,
                    title: $section_title,
                    document: $doc_title{metadata_props}
                }})
                MERGE (c)-[:CONTAINS]->(s)
                SET s.content = $content
            """, **metadata_params)
            
            # Create Benefit nodes
            benefit_costs = extract_benefit_costs(section["content"])
            for benefit, cost in benefit_costs.items():
                session.run("""
                    MATCH (s:Section {number: $section_num, document: $doc_title})
                    MERGE (b:Benefit {
                        name: $benefit,
                        cost: $cost,
                        document: $doc_title
                    })
                    MERGE (s)-[:HAS_BENEFIT]->(b)
                """, section_num=section["section_number"], 
                     doc_title=doc_title,
                     benefit=benefit,
                     cost=cost)
            
            # Create Feature nodes
            features = identify_available_features(section["content"])
            for feature, available in features.items():
                session.run("""
                    MATCH (s:Section {number: $section_num, document: $doc_title})
                    MERGE (f:Feature {
                        name: $feature,
                        available: $available,
                        document: $doc_title
                    })
                    MERGE (s)-[:HAS_FEATURE]->(f)
                """, section_num=section["section_number"],
                     doc_title=doc_title,
                     feature=feature,
                     available=available)
        
        # Create Feature nodes for all document features
        for feature, available in document_features.items():
            if available is not None:  # Skip "Not Found" values (None)
                session.run("""
                    MATCH (d:Document {title: $doc_title})
                    MERGE (f:Feature {
                        name: $feature,
                        available: $available,
                        document: $doc_title
                    })
                    MERGE (d)-[:HAS_FEATURE]->(f)
                """, 
                     doc_title=doc_title,
                     feature=feature,
                     available=available)
        
        # Create entities
        for req in entities["requirements"]:
            session.run("""
                MATCH (c:Chapter {number: $chapter_num, document: $doc_title})
                MERGE (r:Requirement {text: $req_text, document: $doc_title})
                MERGE (c)-[:CONTAINS]->(r)
            """, chapter_num=chapter_data["chapter_number"], req_text=req, doc_title=doc_title)
        
        for area in entities["service_areas"]:
            session.run("""
                MATCH (c:Chapter {number: $chapter_num, document: $doc_title})
                MERGE (a:ServiceArea {name: $area_name, document: $doc_title})
                MERGE (c)-[:COVERS]->(a)
            """, chapter_num=chapter_data["chapter_number"], area_name=area, doc_title=doc_title)
        
        # Add cross-references between sections
        for section in chapter_data["sections"]:
            content = section["content"]
            # Find references to other sections
            ref_pattern = re.compile(r"Chapter (\d+), Section (\d+\.\d+)")
            for match in ref_pattern.finditer(content):
                ref_chapter, ref_section = match.groups()
                session.run("""
                    MATCH (s1:Section {number: $section_num, document: $doc_title})
                    MATCH (s2:Section {number: $ref_section, document: $doc_title})
                    MERGE (s1)-[:REFERENCES]->(s2)
                """, section_num=section["section_number"], 
                     ref_section=ref_section,
                     doc_title=doc_title)
    
    driver.close()

def generate_visualization_query(doc_title):
    """Generate Neo4j query for visualizing document structure"""
    return f"""
    MATCH path = (d:Document {{title: "{doc_title}"}})-[:CONTAINS*1..3]->()
    RETURN path LIMIT 100
    """

def process_medicare_document(pdf_path, neo4j_uri, neo4j_user, neo4j_password, doc_title):
    """Process a Medicare document and store in Neo4j"""
    print(f"Processing document: {doc_title}")
    
    # First try to extract features from summary tables
    found_summary = extract_features_from_summary_tables(pdf_path, doc_title, neo4j_uri, neo4j_user, neo4j_password)
    if found_summary:
        print("Successfully extracted features from summary tables")
    
    # Process each chapter as before
    for chapter_num in range(1, 21):  # Support up to 20 chapters
        print(f"Searching for Chapter {chapter_num}...")
        
        # Parse chapter
        chapter_data = parse_medicare_chapter(pdf_path, chapter_num)
        
        if not chapter_data["sections"]:
            print(f"No sections found for Chapter {chapter_num}")
            continue  # Skip if chapter not found
        
        print(f"Found {len(chapter_data['sections'])} sections in Chapter {chapter_num}")
        
        # Extract entities
        entities = extract_entities(chapter_data)
        
        # Populate Neo4j
        populate_neo4j(neo4j_uri, neo4j_user, neo4j_password, chapter_data, entities, doc_title)
        
        print(f"Processed Chapter {chapter_num}")
    
    print(f"Document processing complete for: {doc_title}")
    print(f"To visualize this document, run this query in Neo4j Browser:")
    print(generate_visualization_query(doc_title))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Medicare document chapters into Neo4j')
    parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    parser.add_argument('--uri', required=True, help='Neo4j connection URI')
    parser.add_argument('--user', required=True, help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--doc-title', required=True, help='Title of the document')
    
    args = parser.parse_args()
    
    process_medicare_document(args.pdf, args.uri, args.user, args.password, args.doc_title)