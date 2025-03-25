from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import spacy
from neo4j import GraphDatabase
import argparse

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

def populate_neo4j(uri, username, password, chapter_data, entities):
    """Populate Neo4j database with chapter data and entities"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # Create Document node if not exists
        session.run("""
            MERGE (d:Document {title: $title})
        """, title="AARP Medicare Advantage Evidence of Coverage")
        
        # Create Chapter node
        session.run("""
            MATCH (d:Document {title: $doc_title})
            MERGE (c:Chapter {
                number: $chapter_num,
                title: $chapter_title
            })
            MERGE (d)-[:CONTAINS]->(c)
        """, doc_title="AARP Medicare Advantage Evidence of Coverage",
             chapter_num=chapter_data["chapter_number"],
             chapter_title=chapter_data["title"])
        
        # Create Section nodes
        for section in chapter_data["sections"]:
            session.run("""
                MATCH (c:Chapter {number: $chapter_num})
                MERGE (s:Section {
                    number: $section_num,
                    title: $section_title
                })
                MERGE (c)-[:CONTAINS]->(s)
                SET s.content = $content
            """, chapter_num=chapter_data["chapter_number"],
                 section_num=section["section_number"],
                 section_title=section["title"],
                 content=section["content"])
        
        # Create entities
        for req in entities["requirements"]:
            session.run("""
                MATCH (c:Chapter {number: $chapter_num})
                MERGE (r:Requirement {text: $req_text})
                MERGE (c)-[:CONTAINS]->(r)
            """, chapter_num=chapter_data["chapter_number"], req_text=req)
        
        for area in entities["service_areas"]:
            session.run("""
                MATCH (c:Chapter {number: $chapter_num})
                MERGE (a:ServiceArea {name: $area_name})
                MERGE (c)-[:COVERS]->(a)
            """, chapter_num=chapter_data["chapter_number"], area_name=area)
        
        # Add cross-references between sections
        for section in chapter_data["sections"]:
            content = section["content"]
            # Find references to other sections
            ref_pattern = re.compile(r"Chapter (\d+), Section (\d+\.\d+)")
            for match in ref_pattern.finditer(content):
                ref_chapter, ref_section = match.groups()
                session.run("""
                    MATCH (s1:Section {number: $section_num})
                    MATCH (s2:Section {number: $ref_section})
                    MERGE (s1)-[:REFERENCES]->(s2)
                """, section_num=section["section_number"], 
                     ref_section=ref_section)
    
    driver.close()

def generate_visualization_query():
    """Generate Neo4j query for visualizing document structure"""
    return """
    MATCH path = (d:Document)-[:CONTAINS*1..3]->()
    RETURN path LIMIT 100
    """

def process_medicare_document(pdf_path, neo4j_uri, neo4j_user, neo4j_password):
    """Process a Medicare document and store in Neo4j"""
    # Process each chapter - INCREASED RANGE
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
        populate_neo4j(neo4j_uri, neo4j_user, neo4j_password, chapter_data, entities)
        
        print(f"Processed Chapter {chapter_num}")
    
    print("Document processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Medicare document chapters into Neo4j')
    parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    parser.add_argument('--uri', required=True, help='Neo4j connection URI')
    parser.add_argument('--user', required=True, help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    
    args = parser.parse_args()
    
    process_medicare_document(args.pdf, args.uri, args.user, args.password)