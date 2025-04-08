# # import os
# # import fitz  # PyMuPDF
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from sentence_transformers import SentenceTransformer
# # from neo4j import GraphDatabase
# # import uuid
# # import spacy
# # import re

# # class GraphRAGPipeline:
# #     def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
# #         """Initialize the GraphRAG pipeline with Neo4j connection details."""
# #         self.neo4j_uri = neo4j_uri
# #         self.neo4j_user = neo4j_user
# #         self.neo4j_password = neo4j_password
        
# #         # Initialize components
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=1000,
# #             chunk_overlap=200,
# #             length_function=len,
# #         )
        
# #         # Initialize embedder
# #         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
# #         # Initialize NER for entity extraction
# #         self.nlp = spacy.load("en_core_web_sm")
        
# #         # Connect to Neo4j
# #         self.driver = GraphDatabase.driver(
# #             neo4j_uri, 
# #             auth=(neo4j_user, neo4j_password)
# #         )
        
# #         # Initialize database with schema
# #         self._init_database()
    
# #     def _init_database(self):
# #         """Initialize the Neo4j database with the necessary schema."""
# #         with self.driver.session() as session:
# #             # Create constraints
# #             session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
# #             session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
# #             session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            
# #             # Create vector index for embeddings if it doesn't exist
# #             try:
# #                 session.run("""
# #                     CALL db.index.vector.createNodeIndex(
# #                         'chunkEmbeddings',
# #                         'Chunk',
# #                         'embedding',
# #                         384,  // Dimension of all-MiniLM-L6-v2 embeddings
# #                         'cosine'
# #                     )
# #                 """)
# #             except:
# #                 # Index might already exist
# #                 pass
    
# #     def document_parser(self, pdf_path):
# #         """Parse a PDF file and extract its text."""
# #         text = ""
# #         try:
# #             doc = fitz.open(pdf_path)
# #             for page in doc:
# #                 text += page.get_text()
            
# #             # Get filename without path and extension
# #             filename = os.path.basename(pdf_path)
# #             doc_name = os.path.splitext(filename)[0]
            
# #             return {
# #                 "id": str(uuid.uuid4()),
# #                 "name": doc_name,
# #                 "path": pdf_path,
# #                 "text": text
# #             }
# #         except Exception as e:
# #             print(f"Error parsing document: {e}")
# #             return None
    
# #     def text_splitter_func(self, document):
# #         """Split document text into chunks."""
# #         if not document:
# #             return []
        
# #         chunks = self.text_splitter.split_text(document["text"])
        
# #         return [{
# #             "id": str(uuid.uuid4()),
# #             "document_id": document["id"],
# #             "document_name": document["name"],
# #             "text": chunk,
# #             "chunk_index": i
# #         } for i, chunk in enumerate(chunks)]
    
# #     def chunk_embedder(self, chunks):
# #         """Create embeddings for each text chunk."""
# #         if not chunks:
# #             return []
        
# #         for chunk in chunks:
# #             # Create embedding
# #             embedding = self.embedder.encode(chunk["text"])
# #             chunk["embedding"] = embedding.tolist()
        
# #         return chunks
    
# #     def entity_relation_extractor(self, chunks):
# #         """Extract entities and relationships from chunks."""
# #         if not chunks:
# #             return [], []
        
# #         entities = {}
# #         relations = []
        
# #         for chunk in chunks:
# #             doc = self.nlp(chunk["text"])
            
# #             # Extract entities
# #             for ent in doc.ents:
# #                 entity_id = f"{ent.text.lower().replace(' ', '_')}_{ent.label_}"
# #                 if entity_id not in entities:
# #                     entities[entity_id] = {
# #                         "id": entity_id,
# #                         "text": ent.text,
# #                         "type": ent.label_
# #                     }
                
# #                 # Create relation between chunk and entity
# #                 relations.append({
# #                     "from_id": chunk["id"],
# #                     "to_id": entity_id,
# #                     "type": "CONTAINS_ENTITY"
# #                 })
                
# #             # Attempt to extract simple subject-verb-object relations
# #             for sent in doc.sents:
# #                 for token in sent:
# #                     if token.dep_ == "ROOT" and token.pos_ == "VERB":
# #                         subjects = [subj for subj in token.children if subj.dep_ in ("nsubj", "nsubjpass")]
# #                         objects = [obj for obj in token.children if obj.dep_ in ("dobj", "pobj")]
                        
# #                         for subj in subjects:
# #                             for obj in objects:
# #                                 # Only create relations if both entities exist
# #                                 subj_span = doc[subj.i:subj.i+1]
# #                                 obj_span = doc[obj.i:obj.i+1]
                                
# #                                 if subj_span.text and obj_span.text:
# #                                     subj_id = f"{subj_span.text.lower().replace(' ', '_')}_ENTITY"
# #                                     obj_id = f"{obj_span.text.lower().replace(' ', '_')}_ENTITY"
                                    
# #                                     # Add entities if they don't exist
# #                                     if subj_id not in entities:
# #                                         entities[subj_id] = {
# #                                             "id": subj_id,
# #                                             "text": subj_span.text,
# #                                             "type": "ENTITY"
# #                                         }
                                    
# #                                     if obj_id not in entities:
# #                                         entities[obj_id] = {
# #                                             "id": obj_id,
# #                                             "text": obj_span.text,
# #                                             "type": "ENTITY"
# #                                         }
                                    
# #                                     # Create relation between entities
# #                                     relations.append({
# #                                         "from_id": subj_id,
# #                                         "to_id": obj_id,
# #                                         "type": token.lemma_.upper(),
# #                                         "text": token.text
# #                                     })
        
# #         return list(entities.values()), relations
    
# #     def kg_writer(self, document, chunks, entities, relations):
# #         """Write document, chunks, entities, and relations to Neo4j."""
# #         with self.driver.session() as session:
# #             # Create document node
# #             session.run("""
# #                 MERGE (d:Document {id: $id})
# #                 SET d.name = $name,
# #                     d.path = $path
# #             """, document)
            
# #             # Create chunk nodes and connect to document
# #             for chunk in chunks:
# #                 session.run("""
# #                     MERGE (c:Chunk {id: $id})
# #                     SET c.text = $text,
# #                         c.chunk_index = $chunk_index,
# #                         c.embedding = $embedding
# #                     WITH c
# #                     MATCH (d:Document {id: $document_id})
# #                     MERGE (d)-[:HAS_CHUNK]->(c)
# #                 """, chunk)
            
# #             # Create entity nodes
# #             for entity in entities:
# #                 session.run("""
# #                     MERGE (e:Entity {id: $id})
# #                     SET e.text = $text,
# #                         e.type = $type
# #                 """, entity)
            
# #             # Create relations
# #             for relation in relations:
# #                 if relation["type"] == "CONTAINS_ENTITY":
# #                     session.run("""
# #                         MATCH (c:Chunk {id: $from_id})
# #                         MATCH (e:Entity {id: $to_id})
# #                         MERGE (c)-[:CONTAINS_ENTITY]->(e)
# #                     """, relation)
# #                 else:
# #                     session.run("""
# #                         MATCH (s:Entity {id: $from_id})
# #                         MATCH (o:Entity {id: $to_id})
# #                         MERGE (s)-[r:`${type}`]->(o)
# #                         SET r.text = $text
# #                     """.replace("`${type}`", f"`{relation['type']}`"), relation)
    
# #     def process_pdf(self, pdf_path):
# #         """Process a PDF through the complete pipeline."""
# #         # 1. Parse document
# #         document = self.document_parser(pdf_path)
# #         if not document:
# #             return False
        
# #         # 2. Split text
# #         chunks = self.text_splitter_func(document)
        
# #         # 3. Create embeddings
# #         chunks = self.chunk_embedder(chunks)
        
# #         # 4. Extract entities and relations
# #         entities, relations = self.entity_relation_extractor(chunks)
        
# #         # 5. Write to Neo4j
# #         self.kg_writer(document, chunks, entities, relations)
        
# #         return True
    
# #     def query_knowledge_graph(self, query, limit=5):
# #         """Query the knowledge graph using vector similarity search."""
# #         # Create embedding for the query
# #         query_embedding = self.embedder.encode(query).tolist()
        
# #         with self.driver.session() as session:
# #             # Vector similarity search
# #             result = session.run("""
# #                 CALL db.index.vector.queryNodes('chunkEmbeddings', $limit, $embedding)
# #                 YIELD node, score
# #                 RETURN node.id AS id, node.text AS text, score
# #                 ORDER BY score DESC
# #             """, {"limit": limit, "embedding": query_embedding})
            
# #             chunks = [{"id": record["id"], "text": record["text"], "score": record["score"]} 
# #                      for record in result]
            
# #             # Get related entities
# #             if chunks:
# #                 chunk_ids = [chunk["id"] for chunk in chunks]
# #                 result = session.run("""
# #                     MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
# #                     WHERE c.id IN $chunk_ids
# #                     RETURN e.id AS id, e.text AS text, e.type AS type, 
# #                            count(*) AS frequency
# #                     ORDER BY frequency DESC
# #                     LIMIT 10
# #                 """, {"chunk_ids": chunk_ids})
                
# #                 entities = [{"id": record["id"], "text": record["text"], 
# #                              "type": record["type"], "frequency": record["frequency"]}
# #                            for record in result]
                
# #                 # Get relationships between entities
# #                 entity_ids = [entity["id"] for entity in entities]
# #                 result = session.run("""
# #                     MATCH (e1:Entity)-[r]->(e2:Entity)
# #                     WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
# #                     RETURN e1.text AS from_text, type(r) AS relation, 
# #                            e2.text AS to_text, count(*) AS frequency
# #                     ORDER BY frequency DESC
# #                     LIMIT 10
# #                 """, {"entity_ids": entity_ids})
                
# #                 relations = [{"from": record["from_text"], 
# #                               "relation": record["relation"],
# #                               "to": record["to_text"],
# #                               "frequency": record["frequency"]}
# #                             for record in result]
                
# #                 return {
# #                     "chunks": chunks,
# #                     "entities": entities,
# #                     "relations": relations
# #                 }
            
# #             return {"chunks": [], "entities": [], "relations": []}
    
# #     def close(self):
# #         """Close the Neo4j connection."""
# #         self.driver.close()


# # # Example usage
# # def main():
# #     # Replace with your Neo4j connection details
# #     neo4j_uri = "bolt://localhost:7687"
# #     neo4j_user = "neo4j"
# #     neo4j_password = "Parser328@"
    
# #     # Initialize the pipeline
# #     pipeline = GraphRAGPipeline(neo4j_uri, neo4j_user, neo4j_password)
    
# #     # Process a PDF
# #     pdf_path = "/Users/pranaligole/MS/Courses /Sem 2/CS 532/Code/EOC_2.pdf"
# #     success = pipeline.process_pdf(pdf_path)
    
# #     if success:
# #         print(f"Successfully processed {pdf_path}")
        
# #         # Example query
# #         query = "What is the main topic of the document?"
# #         results = pipeline.query_knowledge_graph(query)
        
# #         print(f"\nQuery: {query}")
# #         print("\nRelevant chunks:")
# #         for i, chunk in enumerate(results["chunks"]):
# #             print(f"{i+1}. (Score: {chunk['score']:.4f}) {chunk['text'][:200]}...")
        
# #         print("\nEntities:")
# #         for entity in results["entities"]:
# #             print(f"- {entity['text']} ({entity['type']}): {entity['frequency']} occurrences")
        
# #         print("\nRelationships:")
# #         for relation in results["relations"]:
# #             print(f"- {relation['from']} {relation['relation']} {relation['to']}: {relation['frequency']} occurrences")
    
# #     # Close connections
# #     pipeline.close()

# # if __name__ == "__main__":
# #     main()


# import os
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from neo4j import GraphDatabase
# import uuid
# import spacy
# import re

# class GraphRAGPipeline:
#     def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
#         """Initialize the GraphRAG pipeline with Neo4j connection details."""
#         self.neo4j_uri = neo4j_uri
#         self.neo4j_user = neo4j_user
#         self.neo4j_password = neo4j_password
        
#         # Initialize components
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#         )
        
#         # Initialize embedder
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Initialize NER for entity extraction
#         self.nlp = spacy.load("en_core_web_sm")
        
#         # Connect to Neo4j
#         self.driver = GraphDatabase.driver(
#             neo4j_uri, 
#             auth=(neo4j_user, neo4j_password)
#         )
        
#         # Initialize database with schema
#         self._init_database()
    
#     def _init_database(self):
#         """Initialize the Neo4j database with the necessary schema."""
#         with self.driver.session() as session:
#             # Create constraints
#             session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
#             session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
#             session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            
#             # Create vector index for embeddings if it doesn't exist
#             try:
#                 session.run("""
#                     CALL db.index.vector.createNodeIndex(
#                         'chunkEmbeddings',
#                         'Chunk',
#                         'embedding',
#                         384,  // Dimension of all-MiniLM-L6-v2 embeddings
#                         'cosine'
#                     )
#                 """)
#             except:
#                 # Index might already exist
#                 pass
    
#     def document_parser(self, pdf_path):
#         """Parse a PDF file and extract its text."""
#         text = ""
#         try:
#             doc = fitz.open(pdf_path)
#             for page in doc:
#                 text += page.get_text()
            
#             # Get filename without path and extension
#             filename = os.path.basename(pdf_path)
#             doc_name = os.path.splitext(filename)[0]
            
#             return {
#                 "id": str(uuid.uuid4()),
#                 "name": doc_name,
#                 "path": pdf_path,
#                 "text": text
#             }
#         except Exception as e:
#             print(f"Error parsing document: {e}")
#             return None
    
#     def text_splitter_func(self, document):
#         """Split document text into chunks."""
#         if not document:
#             return []
        
#         chunks = self.text_splitter.split_text(document["text"])
        
#         return [{
#             "id": str(uuid.uuid4()),
#             "document_id": document["id"],
#             "document_name": document["name"],
#             "text": chunk,
#             "chunk_index": i
#         } for i, chunk in enumerate(chunks)]
    
#     def chunk_embedder(self, chunks):
#         """Create embeddings for each text chunk."""
#         if not chunks:
#             return []
        
#         for chunk in chunks:
#             # Create embedding
#             embedding = self.embedder.encode(chunk["text"])
#             chunk["embedding"] = embedding.tolist()
        
#         return chunks
    
#     def entity_relation_extractor(self, chunks):
#         """Extract entities and relationships from chunks with improved relation extraction."""
#         if not chunks:
#             return [], []
        
#         entities = {}
#         relations = []
#         document_id = chunks[0]["document_id"] if chunks else None
#         document_name = chunks[0]["document_name"] if chunks else "unknown"
        
#         # Process entities first (across all chunks)
#         for chunk in chunks:
#             doc = self.nlp(chunk["text"])
            
#             # Extract entities
#             for ent in doc.ents:
#                 # Filter out low-quality entities
#                 if len(ent.text.strip()) < 2 or ent.text.strip().isdigit():
#                     continue
                    
#                 # Create document-specific entity ID
#                 entity_id = f"{document_name}_{ent.text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_{ent.label_}"
#                 if entity_id not in entities:
#                     entities[entity_id] = {
#                         "id": entity_id,
#                         "text": ent.text,
#                         "type": ent.label_,
#                         "document_id": document_id,
#                         "document_name": document_name
#                     }
                
#                 # Create relation between chunk and entity
#                 relations.append({
#                     "from_id": chunk["id"],
#                     "to_id": entity_id,
#                     "type": "CONTAINS_ENTITY",
#                     "document_id": document_id,
#                     "document_name": document_name
#                 })
        
#         # Now extract relationships between entities
#         for chunk in chunks:
#             doc = self.nlp(chunk["text"])
            
#             # Create a mapping of token indices to entity IDs for this chunk
#             token_to_entity = {}
#             for ent in doc.ents:
#                 entity_text = ent.text.lower()
#                 for token_idx in range(ent.start, ent.end):
#                     entity_id = f"{document_name}_{entity_text.replace(' ', '_').replace('.', '').replace(',', '')}_{ent.label_}"
#                     if entity_id in entities:
#                         token_to_entity[token_idx] = entity_id
            
#             # Extract relations using dependency parsing patterns
#             for sent in doc.sents:
#                 # Skip sentences that are too short
#                 if len(sent) < 4:
#                     continue
                
#                 # Pattern 1: Subject-Verb-Object triplets
#                 for token in sent:
#                     # Look for verbs that are either the root or part of a verbal phrase
#                     if token.pos_ == "VERB" and (token.dep_ == "ROOT" or token.dep_ in ("ccomp", "xcomp")):
#                         # Find all subject and object candidates
#                         subjects = []
#                         for subj in token.children:
#                             if subj.dep_ in ("nsubj", "nsubjpass"):
#                                 # Get the full subject noun phrase
#                                 subjects.append(self._get_full_span(subj))
                        
#                         objects = []
#                         for obj in token.children:
#                             if obj.dep_ in ("dobj", "pobj", "attr", "iobj"):
#                                 # Get the full object noun phrase
#                                 objects.append(self._get_full_span(obj))
                        
#                         # Create relations between subjects and objects
#                         for subj_span in subjects:
#                             for obj_span in objects:
#                                 if subj_span.text.strip() and obj_span.text.strip():
#                                     # Clean and normalize the texts
#                                     subj_text = subj_span.text.strip()
#                                     obj_text = obj_span.text.strip()
                                    
#                                     # Skip if either subject or object is too short
#                                     if len(subj_text) < 2 or len(obj_text) < 2:
#                                         continue
                                    
#                                     # Create entity IDs with document prefix
#                                     subj_id = f"{document_name}_{subj_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
#                                     obj_id = f"{document_name}_{obj_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
                                    
#                                     # Create meaningful relation type
#                                     relation_type = token.lemma_.upper()
#                                     if len(relation_type) < 2:  # Skip single-letter relation types
#                                         continue
                                    
#                                     # Add entities if they don't exist
#                                     if subj_id not in entities:
#                                         entities[subj_id] = {
#                                             "id": subj_id,
#                                             "text": subj_text,
#                                             "type": "ENTITY",
#                                             "document_id": document_id,
#                                             "document_name": document_name
#                                         }
                                    
#                                     if obj_id not in entities:
#                                         entities[obj_id] = {
#                                             "id": obj_id,
#                                             "text": obj_text,
#                                             "type": "ENTITY",
#                                             "document_id": document_id,
#                                             "document_name": document_name
#                                         }
                                    
#                                     # Create relation between entities with additional metadata
#                                     relations.append({
#                                         "from_id": subj_id,
#                                         "to_id": obj_id,
#                                         "type": relation_type,
#                                         "text": token.text,
#                                         "document_id": document_id,
#                                         "document_name": document_name,
#                                         "sentence": sent.text
#                                     })
                
#                 # Pattern 2: Entity-Preposition-Entity triplets
#                 for token in sent:
#                     if token.pos_ == "ADP" and token.dep_ == "prep":
#                         # Find the head entity and the prepositional object
#                         head = token.head
#                         prep_objects = [child for child in token.children if child.dep_ == "pobj"]
                        
#                         if head and prep_objects:
#                             head_span = self._get_full_span(head)
                            
#                             for pobj in prep_objects:
#                                 pobj_span = self._get_full_span(pobj)
                                
#                                 if head_span.text.strip() and pobj_span.text.strip():
#                                     # Clean and normalize the texts
#                                     head_text = head_span.text.strip()
#                                     pobj_text = pobj_span.text.strip()
                                    
#                                     # Skip if either entity is too short
#                                     if len(head_text) < 2 or len(pobj_text) < 2:
#                                         continue
                                    
#                                     # Create entity IDs with document prefix
#                                     head_id = f"{document_name}_{head_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
#                                     pobj_id = f"{document_name}_{pobj_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
                                    
#                                     # Create relation type from the preposition
#                                     relation_type = f"HAS_{token.text.upper()}"
                                    
#                                     # Add entities if they don't exist
#                                     if head_id not in entities:
#                                         entities[head_id] = {
#                                             "id": head_id,
#                                             "text": head_text,
#                                             "type": "ENTITY",
#                                             "document_id": document_id,
#                                             "document_name": document_name
#                                         }
                                    
#                                     if pobj_id not in entities:
#                                         entities[pobj_id] = {
#                                             "id": pobj_id,
#                                             "text": pobj_text,
#                                             "type": "ENTITY",
#                                             "document_id": document_id,
#                                             "document_name": document_name
#                                         }
                                    
#                                     # Create relation between entities
#                                     relations.append({
#                                         "from_id": head_id,
#                                         "to_id": pobj_id,
#                                         "type": relation_type,
#                                         "text": token.text,
#                                         "document_id": document_id,
#                                         "document_name": document_name,
#                                         "sentence": sent.text
#                                     })
        
#         return list(entities.values()), relations
        
#     def _get_full_span(self, token):
#         """Get the full noun phrase span for a given token."""
#         min_i = token.i
#         max_i = token.i
        
#         # Include all children to get the complete noun phrase
#         children = list(token.children)
#         for child in children:
#             if child.i < min_i:
#                 min_i = child.i
#             if child.i > max_i:
#                 max_i = child.i
#             # Add child's children to the list
#             children.extend(child.children)
        
#         # Return the span from the doc
#         return token.doc[min_i:max_i + 1]
    
#     def kg_writer(self, document, chunks, entities, relations):
#         """Write document, chunks, entities, and relations to Neo4j with document-specific labeling."""
#         # Create a valid Neo4j label from document name
#         doc_label = document["name"].replace(" ", "_").replace("-", "_").replace(".", "_")
        
#         with self.driver.session() as session:
#             # Create document node with document-specific label
#             session.run(f"""
#                 MERGE (d:Document:`{doc_label}` {{id: $id}})
#                 SET d.name = $name,
#                     d.path = $path,
#                     d.display_name = $name
#             """, document)
            
#             # Create chunk nodes and connect to document
#             for chunk in chunks:
#                 session.run(f"""
#                     MERGE (c:Chunk:`{doc_label}` {{id: $id}})
#                     SET c.text = $text,
#                         c.chunk_index = $chunk_index,
#                         c.embedding = $embedding,
#                         c.document_name = $document_name,
#                         c.display_name = substring($text, 0, 30) + '...'
#                     WITH c
#                     MATCH (d:Document {{id: $document_id}})
#                     MERGE (d)-[:HAS_CHUNK]->(c)
#                 """, chunk)
            
#             # Create entity nodes
#             for entity in entities:
#                 session.run(f"""
#                     MERGE (e:Entity:`{doc_label}` {{id: $id}})
#                     SET e.text = $text,
#                         e.type = $type,
#                         e.document_name = $document_name,
#                         e.display_name = $text
#                 """, entity)
            
#             # Create relations
#             for relation in relations:
#                 if relation["type"] == "CONTAINS_ENTITY":
#                     session.run(f"""
#                         MATCH (c:Chunk {{id: $from_id}})
#                         MATCH (e:Entity {{id: $to_id}})
#                         MERGE (c)-[:CONTAINS_ENTITY {{document_name: $document_name}}]->(e)
#                     """, relation)
#                 else:
#                     # Create a valid relationship type
#                     rel_type = relation["type"].replace(" ", "_").replace("-", "_")
#                     if not rel_type:
#                         rel_type = "RELATED_TO"
                        
#                     session.run(f"""
#                         MATCH (s:Entity {{id: $from_id}})
#                         MATCH (o:Entity {{id: $to_id}})
#                         MERGE (s)-[r:`{rel_type}` {{document_name: $document_name}}]->(o)
#                         SET r.text = $text,
#                             r.sentence = $sentence
#                     """, relation)
    
#     def process_pdf(self, pdf_path):
#         """Process a PDF through the complete pipeline."""
#         # 1. Parse document
#         document = self.document_parser(pdf_path)
#         if not document:
#             return False
        
#         # 2. Split text
#         chunks = self.text_splitter_func(document)
        
#         # 3. Create embeddings
#         chunks = self.chunk_embedder(chunks)
        
#         # 4. Extract entities and relations
#         entities, relations = self.entity_relation_extractor(chunks)
        
#         # 5. Write to Neo4j
#         self.kg_writer(document, chunks, entities, relations)
        
#         return True
    
#     def query_knowledge_graph(self, query, document_name=None, limit=5):
#         """Query the knowledge graph using vector similarity search with optional document filter."""
#         # Create embedding for the query
#         query_embedding = self.embedder.encode(query).tolist()
        
#         with self.driver.session() as session:
#             # Vector similarity search with optional document filter
#             if document_name:
#                 # Convert document name to valid label format
#                 doc_label = document_name.replace(" ", "_").replace("-", "_").replace(".", "_")
#                 result = session.run(f"""
#                     CALL db.index.vector.queryNodes('chunkEmbeddings', $limit, $embedding)
#                     YIELD node, score
#                     WHERE node:Chunk:`{doc_label}`
#                     RETURN node.id AS id, node.text AS text, node.document_name AS document_name, score
#                     ORDER BY score DESC
#                 """, {"limit": limit, "embedding": query_embedding})
#             else:
#                 result = session.run("""
#                     CALL db.index.vector.queryNodes('chunkEmbeddings', $limit, $embedding)
#                     YIELD node, score
#                     RETURN node.id AS id, node.text AS text, node.document_name AS document_name, score
#                     ORDER BY score DESC
#                 """, {"limit": limit, "embedding": query_embedding})
            
#             chunks = [{"id": record["id"], "text": record["text"], 
#                       "document_name": record["document_name"], "score": record["score"]} 
#                      for record in result]
            
#             # Get related entities
#             if chunks:
#                 chunk_ids = [chunk["id"] for chunk in chunks]
#                 doc_filter = f"AND e:`{doc_label}`" if document_name else ""
                
#                 query = f"""
#                     MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
#                     WHERE c.id IN $chunk_ids {doc_filter}
#                     RETURN e.id AS id, e.text AS text, e.type AS type, 
#                            e.document_name AS document_name, count(*) AS frequency
#                     ORDER BY frequency DESC
#                     LIMIT 10
#                 """
                
#                 result = session.run(query, {"chunk_ids": chunk_ids})
                
#                 entities = [{"id": record["id"], "text": record["text"], 
#                              "type": record["type"], "document_name": record["document_name"],
#                              "frequency": record["frequency"]}
#                            for record in result]
                
#                 # Get relationships between entities
#                 entity_ids = [entity["id"] for entity in entities]
#                 doc_filter = f"AND type(r) CONTAINS '{document_name}'" if document_name else ""
                
#                 query = f"""
#                     MATCH (e1:Entity)-[r]->(e2:Entity)
#                     WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids {doc_filter}
#                     RETURN e1.text AS from_text, type(r) AS relation, 
#                            e2.text AS to_text, r.document_name AS document_name,
#                            r.sentence AS sentence, count(*) AS frequency
#                     ORDER BY frequency DESC
#                     LIMIT 10
#                 """
                
#                 result = session.run(query, {"entity_ids": entity_ids})
                
#                 relations = [{"from": record["from_text"], 
#                               "relation": record["relation"],
#                               "to": record["to_text"],
#                               "document_name": record["document_name"],
#                               "sentence": record.get("sentence", ""),
#                               "frequency": record["frequency"]}
#                             for record in result]
                
#                 return {
#                     "chunks": chunks,
#                     "entities": entities,
#                     "relations": relations
#                 }
            
#             return {"chunks": [], "entities": [], "relations": []}
            
#     def get_document_graph_queries(self, document_name):
#         """Return Cypher queries to view document-specific graph."""
#         # Convert document name to valid label format
#         doc_label = document_name.replace(" ", "_").replace("-", "_").replace(".", "_")
        
#         queries = {
#             "document_structure": f"""
#                 MATCH (d:Document:`{doc_label}`)-[:HAS_CHUNK]->(c:Chunk)
#                 RETURN d, c LIMIT 100
#             """,
            
#             "document_entities": f"""
#                 MATCH (d:Document:`{doc_label}`)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
#                 RETURN d, c, e LIMIT 100
#             """,
            
#             "top_entities": f"""
#                 MATCH (c:Chunk:`{doc_label}`)-[:CONTAINS_ENTITY]->(e:Entity)
#                 WITH e, count(c) as freq
#                 WHERE freq > 1
#                 RETURN e.text, e.type, freq
#                 ORDER BY freq DESC
#                 LIMIT 20
#             """,
            
#             "entity_relationships": f"""
#                 MATCH (e1:Entity:`{doc_label}`)-[r]->(e2:Entity:`{doc_label}`)
#                 RETURN e1.text AS source, type(r) AS relationship, e2.text AS target
#                 LIMIT 100
#             """
#         }
        
#         return queries
    

    
#     def close(self):
#         """Close the Neo4j connection."""
#         self.driver.close()


# # Example usage
# def main():
#     # Replace with your Neo4j connection details
#     neo4j_uri = "bolt://localhost:7687"
#     neo4j_user = "neo4j"
#     neo4j_password = "Parser328@"
    
#     # Initialize the pipeline
#     pipeline = GraphRAGPipeline(neo4j_uri, neo4j_user, neo4j_password)
    
#     # Process a PDF
#     pdf_path = "EOC_2.pdf"
#     success = pipeline.process_pdf(pdf_path)
    
#     if success:
#         print(f"Successfully processed {pdf_path}")
        
#         # Example query
#         query = "What is the main topic of the document?"
#         results = pipeline.query_knowledge_graph(query)
        
#         print(f"\nQuery: {query}")
#         print("\nRelevant chunks:")
#         for i, chunk in enumerate(results["chunks"]):
#             print(f"{i+1}. (Score: {chunk['score']:.4f}) {chunk['text'][:200]}...")
        
#         print("\nEntities:")
#         for entity in results["entities"]:
#             print(f"- {entity['text']} ({entity['type']}): {entity['frequency']} occurrences")
        
#         print("\nRelationships:")
#         for relation in results["relations"]:
#             print(f"- {relation['from']} {relation['relation']} {relation['to']}: {relation['frequency']} occurrences")
    
#     # Close connections
#     pipeline.close()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
GraphRAG Pipeline Implementation - Updated version without GDS dependency,
calculating vector similarity in Python instead.
"""

import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import uuid
import spacy
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GraphRAGPipeline:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        """Initialize the GraphRAG pipeline with Neo4j connection details."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize NER for entity extraction
        self.nlp = spacy.load("en_core_web_sm")
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        
        # Initialize database with schema
        self._init_database()
    
    def _init_database(self):
        """Initialize the Neo4j database with the necessary schema."""
        with self.driver.session() as session:
            # Create constraints
            try:
                session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create constraints. This might be ok for older Neo4j versions: {e}")
                # Try alternative syntax for older Neo4j versions
                try:
                    session.run("CREATE CONSTRAINT ON (d:Document) ASSERT d.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT ON (c:Chunk) ASSERT c.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE")
                except Exception as e2:
                    print(f"Warning: Could not create constraints with older syntax either: {e2}")
                    print("Continuing without constraints.")
            
            # No vector index creation - we'll use Python-based similarity instead
    
    def document_parser(self, pdf_path):
        """Parse a PDF file and extract its text."""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            
            # Get filename without path and extension
            filename = os.path.basename(pdf_path)
            doc_name = os.path.splitext(filename)[0]
            
            return {
                "id": str(uuid.uuid4()),
                "name": doc_name,
                "path": pdf_path,
                "text": text
            }
        except Exception as e:
            print(f"Error parsing document: {e}")
            return None
    
    def text_splitter_func(self, document):
        """Split document text into chunks."""
        if not document:
            return []
        
        chunks = self.text_splitter.split_text(document["text"])
        
        return [{
            "id": str(uuid.uuid4()),
            "document_id": document["id"],
            "document_name": document["name"],
            "text": chunk,
            "chunk_index": i
        } for i, chunk in enumerate(chunks)]
    
    def chunk_embedder(self, chunks):
        """Create embeddings for each text chunk."""
        if not chunks:
            return []
        
        for chunk in chunks:
            # Create embedding
            embedding = self.embedder.encode(chunk["text"])
            chunk["embedding"] = embedding.tolist()
        
        return chunks
    
    def entity_relation_extractor(self, chunks):
        """Extract entities and relationships from chunks with improved relation extraction."""
        if not chunks:
            return [], []
        
        entities = {}
        relations = []
        document_id = chunks[0]["document_id"] if chunks else None
        document_name = chunks[0]["document_name"] if chunks else "unknown"
        
        # Process entities first (across all chunks)
        for chunk in chunks:
            doc = self.nlp(chunk["text"])
            
            # Extract entities
            for ent in doc.ents:
                # Filter out low-quality entities
                if len(ent.text.strip()) < 2 or ent.text.strip().isdigit():
                    continue
                    
                # Create document-specific entity ID
                entity_id = f"{document_name}_{ent.text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_{ent.label_}"
                if entity_id not in entities:
                    entities[entity_id] = {
                        "id": entity_id,
                        "text": ent.text,
                        "type": ent.label_,
                        "document_id": document_id,
                        "document_name": document_name
                    }
                
                # Create relation between chunk and entity
                relations.append({
                    "from_id": chunk["id"],
                    "to_id": entity_id,
                    "type": "CONTAINS_ENTITY",
                    "document_id": document_id,
                    "document_name": document_name
                })
        
        # Now extract relationships between entities
        for chunk in chunks:
            doc = self.nlp(chunk["text"])
            
            # Create a mapping of token indices to entity IDs for this chunk
            token_to_entity = {}
            for ent in doc.ents:
                entity_text = ent.text.lower()
                for token_idx in range(ent.start, ent.end):
                    entity_id = f"{document_name}_{entity_text.replace(' ', '_').replace('.', '').replace(',', '')}_{ent.label_}"
                    if entity_id in entities:
                        token_to_entity[token_idx] = entity_id
            
            # Extract relations using dependency parsing patterns
            for sent in doc.sents:
                # Skip sentences that are too short
                if len(sent) < 4:
                    continue
                
                # Pattern 1: Subject-Verb-Object triplets
                for token in sent:
                    # Look for verbs that are either the root or part of a verbal phrase
                    if token.pos_ == "VERB" and (token.dep_ == "ROOT" or token.dep_ in ("ccomp", "xcomp")):
                        # Find all subject and object candidates
                        subjects = []
                        for subj in token.children:
                            if subj.dep_ in ("nsubj", "nsubjpass"):
                                # Get the full subject noun phrase
                                subjects.append(self._get_full_span(subj))
                        
                        objects = []
                        for obj in token.children:
                            if obj.dep_ in ("dobj", "pobj", "attr", "iobj"):
                                # Get the full object noun phrase
                                objects.append(self._get_full_span(obj))
                        
                        # Create relations between subjects and objects
                        for subj_span in subjects:
                            for obj_span in objects:
                                if subj_span.text.strip() and obj_span.text.strip():
                                    # Clean and normalize the texts
                                    subj_text = subj_span.text.strip()
                                    obj_text = obj_span.text.strip()
                                    
                                    # Skip if either subject or object is too short
                                    if len(subj_text) < 2 or len(obj_text) < 2:
                                        continue
                                    
                                    # Create entity IDs with document prefix
                                    subj_id = f"{document_name}_{subj_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
                                    obj_id = f"{document_name}_{obj_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
                                    
                                    # Create meaningful relation type
                                    relation_type = token.lemma_.upper()
                                    if len(relation_type) < 2:  # Skip single-letter relation types
                                        continue
                                    
                                    # Add entities if they don't exist
                                    if subj_id not in entities:
                                        entities[subj_id] = {
                                            "id": subj_id,
                                            "text": subj_text,
                                            "type": "ENTITY",
                                            "document_id": document_id,
                                            "document_name": document_name
                                        }
                                    
                                    if obj_id not in entities:
                                        entities[obj_id] = {
                                            "id": obj_id,
                                            "text": obj_text,
                                            "type": "ENTITY",
                                            "document_id": document_id,
                                            "document_name": document_name
                                        }
                                    
                                    # Create relation between entities with additional metadata
                                    relations.append({
                                        "from_id": subj_id,
                                        "to_id": obj_id,
                                        "type": relation_type,
                                        "text": token.text,
                                        "document_id": document_id,
                                        "document_name": document_name,
                                        "sentence": sent.text
                                    })
                
                # Pattern 2: Entity-Preposition-Entity triplets
                for token in sent:
                    if token.pos_ == "ADP" and token.dep_ == "prep":
                        # Find the head entity and the prepositional object
                        head = token.head
                        prep_objects = [child for child in token.children if child.dep_ == "pobj"]
                        
                        if head and prep_objects:
                            head_span = self._get_full_span(head)
                            
                            for pobj in prep_objects:
                                pobj_span = self._get_full_span(pobj)
                                
                                if head_span.text.strip() and pobj_span.text.strip():
                                    # Clean and normalize the texts
                                    head_text = head_span.text.strip()
                                    pobj_text = pobj_span.text.strip()
                                    
                                    # Skip if either entity is too short
                                    if len(head_text) < 2 or len(pobj_text) < 2:
                                        continue
                                    
                                    # Create entity IDs with document prefix
                                    head_id = f"{document_name}_{head_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
                                    pobj_id = f"{document_name}_{pobj_text.lower().replace(' ', '_').replace('.', '').replace(',', '')}_ENTITY"
                                    
                                    # Create relation type from the preposition
                                    relation_type = f"HAS_{token.text.upper()}"
                                    
                                    # Add entities if they don't exist
                                    if head_id not in entities:
                                        entities[head_id] = {
                                            "id": head_id,
                                            "text": head_text,
                                            "type": "ENTITY",
                                            "document_id": document_id,
                                            "document_name": document_name
                                        }
                                    
                                    if pobj_id not in entities:
                                        entities[pobj_id] = {
                                            "id": pobj_id,
                                            "text": pobj_text,
                                            "type": "ENTITY",
                                            "document_id": document_id,
                                            "document_name": document_name
                                        }
                                    
                                    # Create relation between entities
                                    relations.append({
                                        "from_id": head_id,
                                        "to_id": pobj_id,
                                        "type": relation_type,
                                        "text": token.text,
                                        "document_id": document_id,
                                        "document_name": document_name,
                                        "sentence": sent.text
                                    })
        
        return list(entities.values()), relations
        
    def _get_full_span(self, token):
        """Get the full noun phrase span for a given token."""
        min_i = token.i
        max_i = token.i
        
        # Include all children to get the complete noun phrase
        children = list(token.children)
        for child in children:
            if child.i < min_i:
                min_i = child.i
            if child.i > max_i:
                max_i = child.i
            # Add child's children to the list
            children.extend(child.children)
        
        # Return the span from the doc
        return token.doc[min_i:max_i + 1]
    
    def kg_writer(self, document, chunks, entities, relations):
        """Write document, chunks, entities, and relations to Neo4j with document-specific labeling."""
        # Create a valid Neo4j label from document name
        doc_label = document["name"].replace(" ", "_").replace("-", "_").replace(".", "_")
        
        with self.driver.session() as session:
            # Create document node with document-specific label
            session.run(f"""
                MERGE (d:Document:`{doc_label}` {{id: $id}})
                SET d.name = $name,
                    d.path = $path,
                    d.display_name = $name
            """, document)
            
            # Create chunk nodes and connect to document
            for chunk in chunks:
                session.run(f"""
                    MERGE (c:Chunk:`{doc_label}` {{id: $id}})
                    SET c.text = $text,
                        c.chunk_index = $chunk_index,
                        c.embedding = $embedding,
                        c.document_name = $document_name,
                        c.display_name = substring($text, 0, 30) + '...'
                    WITH c
                    MATCH (d:Document {{id: $document_id}})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, chunk)
            
            # Create entity nodes
            for entity in entities:
                session.run(f"""
                    MERGE (e:Entity:`{doc_label}` {{id: $id}})
                    SET e.text = $text,
                        e.type = $type,
                        e.document_name = $document_name,
                        e.display_name = $text
                """, entity)
            
            # Create relations
            for relation in relations:
                if relation["type"] == "CONTAINS_ENTITY":
                    session.run(f"""
                        MATCH (c:Chunk {{id: $from_id}})
                        MATCH (e:Entity {{id: $to_id}})
                        MERGE (c)-[:CONTAINS_ENTITY {{document_name: $document_name}}]->(e)
                    """, relation)
                else:
                    # Create a valid relationship type
                    rel_type = relation["type"].replace(" ", "_").replace("-", "_")
                    if not rel_type:
                        rel_type = "RELATED_TO"
                        
                    session.run(f"""
                        MATCH (s:Entity {{id: $from_id}})
                        MATCH (o:Entity {{id: $to_id}})
                        MERGE (s)-[r:`{rel_type}` {{document_name: $document_name}}]->(o)
                        SET r.text = $text,
                            r.sentence = $sentence
                    """, relation)
    
    def process_pdf(self, pdf_path):
        """Process a PDF through the complete pipeline."""
        # 1. Parse document
        document = self.document_parser(pdf_path)
        if not document:
            return False
        
        # 2. Split text
        chunks = self.text_splitter_func(document)
        
        # 3. Create embeddings
        chunks = self.chunk_embedder(chunks)
        
        # 4. Extract entities and relations
        entities, relations = self.entity_relation_extractor(chunks)
        
        # 5. Write to Neo4j
        self.kg_writer(document, chunks, entities, relations)
        
        return True
    
    def query_knowledge_graph(self, query, document_name=None, limit=5):
        """Query the knowledge graph without using GDS vector similarity."""
        # Create embedding for the query
        query_embedding = self.embedder.encode(query).tolist()
        
        with self.driver.session() as session:
            # Without GDS library, we need an alternative approach
            if document_name:
                # First, get chunks for the specific document
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.document_name = $document_name
                    RETURN c.id AS id, c.text AS text, c.document_name AS document_name, 
                           c.embedding AS embedding
                    LIMIT 100
                """, {"document_name": document_name})
                
                # Process manually - calculate cosine similarity in Python
                chunks_with_embeddings = [
                    {
                        "id": record["id"], 
                        "text": record["text"],
                        "document_name": record["document_name"],
                        "embedding": record["embedding"]
                    } 
                    for record in result
                ]
                
                # Calculate similarity scores
                if not chunks_with_embeddings:
                    return {"chunks": [], "entities": [], "relations": []}
                    
                # Convert embeddings to numpy arrays
                chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity_scores = cosine_similarity(query_embedding_np, chunk_embeddings)[0]
                
                # Add scores to chunks and sort
                for i, chunk in enumerate(chunks_with_embeddings):
                    chunk["score"] = float(similarity_scores[i])
                
                # Sort by score and limit results
                chunks = sorted(chunks_with_embeddings, key=lambda x: x["score"], reverse=True)[:limit]
                
                # Remove embeddings from results to avoid clutter
                for chunk in chunks:
                    chunk.pop("embedding", None)
                    
            else:
                # For all documents - using a similar approach
                result = session.run("""
                    MATCH (c:Chunk)
                    RETURN c.id AS id, c.text AS text, c.document_name AS document_name, 
                           c.embedding AS embedding
                    LIMIT 100
                """)
                
                # Process manually as above
                chunks_with_embeddings = [
                    {
                        "id": record["id"], 
                        "text": record["text"],
                        "document_name": record.get("document_name", "unknown"),
                        "embedding": record["embedding"]
                    } 
                    for record in result
                ]
                
                # Calculate similarity scores
                if not chunks_with_embeddings:
                    return {"chunks": [], "entities": [], "relations": []}
                    
                # Convert embeddings to numpy arrays
                chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity_scores = cosine_similarity(query_embedding_np, chunk_embeddings)[0]
                
                # Add scores to chunks and sort
                for i, chunk in enumerate(chunks_with_embeddings):
                    chunk["score"] = float(similarity_scores[i])
                
                # Sort by score and limit results
                chunks = sorted(chunks_with_embeddings, key=lambda x: x["score"], reverse=True)[:limit]
                
                # Remove embeddings from results to avoid clutter
                for chunk in chunks:
                    chunk.pop("embedding", None)
            
            # Get related entities from the selected chunks
            if chunks:
                chunk_ids = [chunk["id"] for chunk in chunks]
                doc_filter = f"AND e.document_name = '{document_name}'" if document_name else ""
                
                query = f"""
                    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    WHERE c.id IN $chunk_ids {doc_filter}
                    RETURN e.id AS id, e.text AS text, e.type AS type, 
                           e.document_name AS document_name, count(*) AS frequency
                    ORDER BY frequency DESC
                    LIMIT 10
                """
                
                result = session.run(query, {"chunk_ids": chunk_ids})
                
                entities = [{"id": record["id"], "text": record["text"], 
                             "type": record["type"], 
                             "document_name": record.get("document_name", "unknown"),
                             "frequency": record["frequency"]}
                           for record in result]
                
                # Get relationships between entities
                entity_ids = [entity["id"] for entity in entities]
                doc_filter = f"AND r.document_name = '{document_name}'" if document_name else ""
                
                query = f"""
                    MATCH (e1:Entity)-[r]->(e2:Entity)
                    WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids {doc_filter}
                    RETURN e1.text AS from_text, type(r) AS relation, 
                           e2.text AS to_text, r.document_name AS document_name,
                           r.sentence AS sentence, count(*) AS frequency
                    ORDER BY frequency DESC
                    LIMIT 10
                """
                
                result = session.run(query, {"entity_ids": entity_ids})
                
                relations = [{"from": record["from_text"], 
                              "relation": record["relation"],
                              "to": record["to_text"],
                              "document_name": record.get("document_name", "unknown"),
                              "sentence": record.get("sentence", ""),
                              "frequency": record["frequency"]}
                            for record in result]
                
                return {
                    "chunks": chunks,
                    "entities": entities,
                    "relations": relations
                }
            
            return {"chunks": [], "entities": [], "relations": []}
    
    def chat_with_document(self, query, document_name=None, model_name="llama3", temperature=0.7, max_tokens=1000):
        """
        Complete RAG implementation: Retrieve from knowledge graph and generate answer using Ollama LLM.
        
        Args:
            query: User question about the document
            document_name: Optional name of document to focus on
            model_name: Ollama model to use (e.g., "llama3", "mistral", "gemma")
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing LLM response and supporting evidence
        """
        import requests
        import json
        
        # Step 1: Retrieve relevant information from knowledge graph
        retrieved_info = self.query_knowledge_graph(query, document_name=document_name, limit=5)
        
        if not retrieved_info["chunks"]:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "chunks": [],
                "entities": [],
                "relations": []
            }
        
        # Step 2: Format retrieved information as context
        context = self._format_context_for_llm(retrieved_info)
        
        # Step 3: Prepare prompt with instructions, context and query
        prompt = f"""You are a helpful assistant that answers questions based on specific document content.
        
CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

Based on the provided context from the document, please answer the question. If the answer is not contained in the context, say "I don't have enough information to answer this question." Include relevant details from the context to support your answer, but be concise. Don't mention that you're using "context" or "chunks" in your answer.
"""

        # Step 4: Call Ollama API to generate response
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            
            if response.status_code == 200:
                # Parse streaming response from Ollama
                full_response = ""
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            response_data = json.loads(line)
                            if "response" in response_data:
                                full_response += response_data["response"]
                        except:
                            pass
            else:
                full_response = f"Error: Failed to get response from Ollama. Status code: {response.status_code}"
        except Exception as e:
            full_response = f"Error communicating with Ollama: {str(e)}"
            
        # Step 5: Return response along with supporting evidence
        return {
            "answer": full_response,
            "chunks": retrieved_info["chunks"],
            "entities": retrieved_info["entities"],
            "relations": retrieved_info["relations"]
        }
    
    def _format_context_for_llm(self, retrieved_info):
        """Format retrieved information as context for the LLM."""
        # Format text chunks
        chunks_text = ""
        for i, chunk in enumerate(retrieved_info["chunks"]):
            chunks_text += f"CHUNK {i+1} (DOCUMENT: {chunk.get('document_name', 'unknown')}): {chunk['text']}\n\n"
        
        # Format entities
        entities_text = "KEY ENTITIES:\n"
        for entity in retrieved_info["entities"]:
            entities_text += f"- {entity['text']} (Type: {entity['type']})\n"
        
        # Format relationships
        relations_text = "RELATIONSHIPS BETWEEN ENTITIES:\n"
        for relation in retrieved_info["relations"]:
            sentence = relation.get("sentence", "")
            if sentence:
                relations_text += f"- {relation['from']} {relation['relation']} {relation['to']}: \"{sentence}\"\n"
            else:
                relations_text += f"- {relation['from']} {relation['relation']} {relation['to']}\n"
        
        # Combine all context elements
        context = f"{chunks_text}\n{entities_text}\n{relations_text}"
        return context
            
    def get_document_graph_queries(self, document_name):
        """Return Cypher queries to view document-specific graph."""
        # Convert document name to valid label format
        doc_label = document_name.replace(" ", "_").replace("-", "_").replace(".", "_")
        
        queries = {
            "document_structure": f"""
                MATCH (d:Document:`{doc_label}`)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d, c LIMIT 100
            """,
            
            "document_entities": f"""
                MATCH (d:Document:`{doc_label}`)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN d, c, e LIMIT 100
            """,
            
            "top_entities": f"""
                MATCH (c:Chunk:`{doc_label}`)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH e, count(c) as freq
                WHERE freq > 1
                RETURN e.text, e.type, freq
                ORDER BY freq DESC
                LIMIT 20
            """,
            
            "entity_relationships": f"""
                MATCH (e1:Entity:`{doc_label}`)-[r]->(e2:Entity:`{doc_label}`)
                RETURN e1.text AS source, type(r) AS relationship, e2.text AS target
                LIMIT 100
            """
        }
        
        return queries
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()


# Example usage
def main():
    # Replace with your Neo4j connection details
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Parser328@"
    
    # Initialize the pipeline
    pipeline = GraphRAGPipeline(neo4j_uri, neo4j_user, neo4j_password)
    
    # Process a PDF
    pdf_path = "EOC_1.pdf"
    success = pipeline.process_pdf(pdf_path)
    
    if success:
        print(f"Successfully processed {pdf_path}")
        
        # Get the document name from the file path
        import os
        document_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Print Cypher queries to visualize this document's graph
        print("\nUse these queries in Neo4j Browser to visualize the graph:")
        queries = pipeline.get_document_graph_queries(document_name)
        for query_name, query in queries.items():
            print(f"\n{query_name}:\n{query}")
        
        # Example questions
        questions = [
            "What is the main topic of the document?",
            "What key entities are mentioned in the document?",
            "What benefits are described in the document?"
        ]
        
        # Use Ollama to answer questions about the document
        for question in questions:
            print(f"\n\nQuestion: {question}")
            
            # Check if Ollama is available, otherwise just show retrieval results
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    # Ollama is available, use RAG with LLM
                    results = pipeline.chat_with_document(question, document_name=document_name)
                    
                    print("\nAnswer:")
                    print(results["answer"])
                    
                    print("\nSupporting chunks:")
                    for i, chunk in enumerate(results["chunks"][:2]):
                        print(f"{i+1}. (Score: {chunk['score']:.4f}) {chunk['text'][:150]}...")
                else:
                    # Ollama not available, just show retrieval
                    results = pipeline.query_knowledge_graph(question, document_name=document_name)
                    
                    print("\nRelevant chunks:")
                    for i, chunk in enumerate(results["chunks"]):
                        print(f"{i+1}. (Score: {chunk['score']:.4f}) {chunk['text'][:150]}...")
            except:
                # Ollama not available, just show retrieval
                results = pipeline.query_knowledge_graph(question, document_name=document_name)
                
                print("\nOllama not available. Install Ollama for complete RAG.")
                print("\nRelevant chunks:")
                for i, chunk in enumerate(results["chunks"]):
                    print(f"{i+1}. (Score: {chunk['score']:.4f}) {chunk['text'][:150]}...")
    
    # Close connections
    pipeline.close()

if __name__ == "__main__":
    main()