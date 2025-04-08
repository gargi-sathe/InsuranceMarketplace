# #!/usr/bin/env python3
"""
Interactive Process PDF - A command-line tool to process PDFs and build a knowledge graph in Neo4j.
This version prompts for PDF path if not provided via command line.
"""

import os
import sys
import argparse
from getpass import getpass
import datetime

# Import the GraphRAGPipeline
try:
    from pipeline import GraphRAGPipeline
except ImportError:
    print("Error: Could not import GraphRAGPipeline.")
    print("Make sure the pipeline.py file is in the same directory or in your PYTHONPATH.")
    sys.exit(1)


def list_existing_documents(pipeline):
    """List all existing documents in the database with details."""
    try:
        with pipeline.driver.session() as session:
            # Try to get detailed information including creation date if available
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e)
                RETURN d.name AS name, 
                       d.path AS path,
                       count(DISTINCT c) AS chunks, 
                       count(DISTINCT e) AS entities
                ORDER BY name
            """)
            
            documents = [
                {
                    "name": record["name"],
                    "path": record["path"],
                    "chunks": record["chunks"],
                    "entities": record["entities"]
                }
                for record in result
            ]
            
            if not documents:
                print("No documents found in the database.")
                return []
            
            print("\n📚 Existing documents in the database:")
            print("─" * 80)
            print(f"{'Document Name':<30} {'Path':<30} {'Chunks':<10} {'Entities':<10}")
            print("─" * 80)
            
            for doc in documents:
                print(f"{doc['name']:<30} {doc.get('path', 'N/A'):<30} {doc['chunks']:<10} {doc['entities']:<10}")
            
            print("─" * 80)
            return documents
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        return []


def get_document_details(pipeline, doc_name):
    """Get detailed information about an existing document."""
    try:
        with pipeline.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {name: $name})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e)
                WITH d, count(DISTINCT c) AS chunks, count(DISTINCT e) AS entities
                RETURN d.name AS name, 
                       d.path AS path,
                       chunks, 
                       entities
            """, {"name": doc_name})
            
            record = result.single()
            if not record:
                return None
            
            return {
                "name": record["name"],
                "path": record["path"],
                "chunks": record["chunks"],
                "entities": record["entities"]
            }
    except Exception as e:
        print(f"⚠️ Warning: Could not get document details: {str(e)}")
        return None


def process_pdfs(pdf_paths, uri, user, password, clear_existing=False, list_only=False, force=False):
    """Process multiple PDFs and build a knowledge graph in Neo4j."""
    # Initialize the pipeline
    try:
        pipeline = GraphRAGPipeline(uri, user, password)
        print(f"✅ Connected to Neo4j at {uri}")
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {str(e)}")
        return False
    
    # List existing documents if requested
    existing_documents = list_existing_documents(pipeline)
    
    # If list_only flag is set, just list documents and exit
    if list_only:
        pipeline.close()
        return True
    
    # If no PDF paths were provided, prompt the user for input
    if not pdf_paths:
        pdf_paths = prompt_for_pdfs()
        if not pdf_paths:
            pipeline.close()
            return False
    
    # Clear existing data if requested
    if clear_existing:
        if existing_documents:
            confirm = input("\n⚠️ WARNING: This will delete ALL existing documents. Continue? (y/n): ").lower()
            if confirm != 'y':
                print("Operation cancelled.")
                pipeline.close()
                return False
                
        try:
            with pipeline.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Cleared existing data from the database")
            # Refresh the existing documents list
            existing_documents = []
        except Exception as e:
            print(f"❌ Error clearing database: {str(e)}")
            return False
    
    # Create a lookup dictionary for existing documents
    existing_doc_names = {doc["name"]: doc for doc in existing_documents}
    
    # Process each PDF
    successful = 0
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            continue
        
        try:
            print(f"\n📄 Processing: {pdf_path}")
            
            # Get document name from file path
            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Generate a unique timestamp suffix if needed
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if document already exists
            if doc_name in existing_doc_names:
                existing_doc = existing_doc_names[doc_name]
                print(f"⚠️ Document '{doc_name}' already exists with:")
                print(f"  - Path: {existing_doc.get('path', 'N/A')}")
                print(f"  - Chunks: {existing_doc['chunks']}")
                print(f"  - Entities: {existing_doc['entities']}")
                
                if force:
                    print("Force flag is set, overwriting existing document.")
                    choice = 'y'
                else:
                    choice = input(f"Do you want to reprocess this document? (y/n/u): ").lower()
                    # 'u' option for unique - adds a timestamp to create a new document
                
                if choice == 'n':
                    print(f"⏭️ Skipping: {pdf_path}")
                    continue
                elif choice == 'u':
                    # Create a new document with timestamp to avoid overwriting
                    doc_name = f"{doc_name}_{timestamp}"
                    print(f"📝 Creating a new document with name: {doc_name}")
                else:
                    # Delete existing document
                    try:
                        with pipeline.driver.session() as session:
                            session.run("""
                                MATCH (d:Document {name: $name})
                                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c)
                                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e)
                                DETACH DELETE d, c, e
                            """, {"name": doc_name})
                        print(f"🗑️ Removed existing document: {doc_name}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not delete existing document: {str(e)}")
                        print("Continuing with processing...")
            
            # Create a modified document object with the potentially updated name
            modified_pdf_path = pdf_path
            
            # Process the PDF
            print(f"⏳ Processing document as '{doc_name}'...")
            success = pipeline.process_pdf(modified_pdf_path)
            
            if success:
                print(f"✅ Successfully processed: {pdf_path}")
                
                # Get document statistics
                doc_details = get_document_details(pipeline, doc_name)
                if doc_details:
                    print(f"   📊 Document details:")
                    print(f"   - Name: {doc_details['name']}")
                    print(f"   - Path: {doc_details['path']}")
                    print(f"   - Chunks: {doc_details['chunks']}")
                    print(f"   - Entities: {doc_details['entities']}")
                
                # Add to existing documents list to catch duplicates in the same batch
                existing_doc_names[doc_name] = {
                    "name": doc_name,
                    "path": modified_pdf_path,
                    "chunks": doc_details["chunks"] if doc_details else 0,
                    "entities": doc_details["entities"] if doc_details else 0
                }
                
                successful += 1
            else:
                print(f"❌ Failed to process: {pdf_path}")
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
    
    # Close the connection
    pipeline.close()
    
    # Print summary
    print(f"\n📊 Summary: Successfully processed {successful} out of {len(pdf_paths)} PDFs")
    
    # Print visualization queries for the processed documents
    if successful > 0:
        print("\n🔍 Neo4j visualization queries:")
        print("To view all documents:")
        print("  MATCH (d:Document) RETURN d")
        print("\nTo view a specific document's knowledge graph:")
        print("  MATCH (d:Document {name: 'DOCUMENT_NAME'})-[:HAS_CHUNK]->(c)-[:CONTAINS_ENTITY]->(e) RETURN d, c, e LIMIT 100")
        print("  (Replace 'DOCUMENT_NAME' with the actual document name)")
    
    return successful > 0


def prompt_for_pdfs():
    """Prompt the user to enter PDF paths interactively."""
    pdf_paths = []
    
    print("\nEnter the paths to PDF files you want to process (one per line).")
    print("Press Enter on an empty line when you're finished.")
    
    while True:
        path = input("PDF path (or Enter to finish): ").strip()
        if not path:
            break
            
        # Expand user directory if needed (e.g., ~/Documents/file.pdf)
        expanded_path = os.path.expanduser(path)
        
        # Check if file exists
        if not os.path.exists(expanded_path):
            print(f"⚠️ Warning: File not found: {expanded_path}")
            choice = input("Add anyway? (y/n): ").lower()
            if choice != 'y':
                continue
                
        pdf_paths.append(expanded_path)
    
    if not pdf_paths:
        print("No PDF paths entered.")
        
    return pdf_paths


def main():
    """Parse arguments and run PDF processing."""
    parser = argparse.ArgumentParser(description="Process PDFs and build a knowledge graph in Neo4j")
    
    parser.add_argument("pdf_paths", nargs='*', help="Paths to PDF files to process")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI (default: bolt://localhost:7687)")
    parser.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
    parser.add_argument("--password", help="Neo4j password (will prompt if not provided)")
    parser.add_argument("--clear", action="store_true", help="Clear all existing data before processing")
    parser.add_argument("--list", action="store_true", help="List existing documents without processing")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing documents without prompting")
    
    args = parser.parse_args()
    
    # Prompt for password if not provided
    password = args.password
    if not password:
        password = getpass("Enter Neo4j password: ")
    
    # If list flag is set, just list documents
    if args.list:
        process_pdfs([], args.uri, args.user, password, list_only=True)
        return
    
    # Process the PDFs (will prompt for paths if none provided)
    process_pdfs(args.pdf_paths, args.uri, args.user, password, args.clear, False, args.force)


if __name__ == "__main__":
    main()