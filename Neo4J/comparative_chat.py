#!/usr/bin/env python3
"""
Comparative Chat with PDFs - A command-line tool to ask comparative questions about multiple PDF documents
using a Neo4j knowledge graph and Ollama LLM.
"""

import os
import sys
import argparse
from getpass import getpass
from neo4j import GraphDatabase
from typing import List, Dict, Any

# Import the GraphRAGPipeline
try:
    from pipeline import GraphRAGPipeline
except ImportError:
    print("Error: Could not import GraphRAGPipeline.")
    print("Make sure the pipeline.py file is in the same directory or in your PYTHONPATH.")
    sys.exit(1)


class ComparativePDFChatCLI:
    """Command-line interface for comparing and chatting with PDF documents."""
    
    def __init__(self):
        self.pipeline = None
        self.available_documents = []
        self.selected_documents = []
        self.comparison_mode = False
        self.ollama_available = False
        self.available_models = []
        self.current_model = "llama3"  # Default model
    
    def setup_connection(self, uri, user, password):
        """Connect to Neo4j and initialize the pipeline."""
        try:
            self.pipeline = GraphRAGPipeline(uri, user, password)
            print("✅ Connected to Neo4j database.")
            return True
        except Exception as e:
            print(f"❌ Error connecting to Neo4j: {str(e)}")
            return False
    
    def check_ollama(self):
        """Check if Ollama is available and get available models."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.ollama_available = True
                models = response.json().get("models", [])
                self.available_models = [model.get("name") for model in models]
                if self.available_models:
                    self.current_model = self.available_models[0]  # Set first model as default
                print(f"✅ Ollama is available with {len(self.available_models)} models.")
            else:
                print("❌ Ollama API returned an error.")
                self.ollama_available = False
        except Exception as e:
            print(f"❌ Ollama is not available: {str(e)}")
            print("📝 Install Ollama from https://ollama.ai for better answers.")
            self.ollama_available = False
    
    def get_available_documents(self):
        """Get list of available documents in the database."""
        try:
            with self.pipeline.driver.session() as session:
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
                
                self.available_documents = [
                    {
                        "name": record["name"],
                        "path": record["path"],
                        "chunks": record["chunks"], 
                        "entities": record["entities"]
                    }
                    for record in result
                ]
                
                if not self.available_documents:
                    print("No documents found in the database.")
                else:
                    print(f"Found {len(self.available_documents)} documents:")
                    for i, doc in enumerate(self.available_documents):
                        print(f"  {i+1}. {doc['name']} ({doc['chunks']} chunks, {doc['entities']} entities)")
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            # Try alternative query for older Neo4j versions
            try:
                with self.pipeline.driver.session() as session:
                    result = session.run("""
                        MATCH (d:Document)
                        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c)
                        OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e)
                        RETURN d.name AS name, d.path as path,
                               count(c) AS chunks, count(e) AS entities
                        ORDER BY name
                    """)
                    
                    self.available_documents = [
                        {
                            "name": record["name"],
                            "path": record["path"],
                            "chunks": record["chunks"], 
                            "entities": record["entities"]
                        }
                        for record in result
                    ]
                    
                    if not self.available_documents:
                        print("No documents found in the database.")
                    else:
                        print(f"Found {len(self.available_documents)} documents:")
                        for i, doc in enumerate(self.available_documents):
                            print(f"  {i+1}. {doc['name']} ({doc['chunks']} chunks, {doc['entities']} entities)")
            except Exception as e2:
                print(f"Error with alternative query: {str(e2)}")
    
    def select_single_document(self):
        """Let user select a single document to chat with."""
        if not self.available_documents:
            print("No documents available. Please process a PDF first.")
            return False
            
        print("\nSelect a document to chat with:")
        for i, doc in enumerate(self.available_documents):
            print(f"  {i+1}. {doc['name']} ({doc['chunks']} chunks, {doc['entities']} entities)")
        
        while True:
            try:
                choice = input("\nEnter document number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return False
                    
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_documents):
                    self.selected_documents = [self.available_documents[idx]["name"]]
                    self.comparison_mode = False
                    print(f"Selected document: {self.selected_documents[0]}")
                    return True
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def select_documents_for_comparison(self):
        """Let user select multiple documents for comparison."""
        if len(self.available_documents) < 2:
            print("You need at least 2 documents to use comparison mode.")
            return False
            
        print("\nSelect documents to compare (at least 2):")
        for i, doc in enumerate(self.available_documents):
            print(f"  {i+1}. {doc['name']} ({doc['chunks']} chunks, {doc['entities']} entities)")
        
        selected_indices = []
        
        while True:
            try:
                if selected_indices:
                    print(f"Currently selected: {', '.join(self.available_documents[idx]['name'] for idx in selected_indices)}")
                    
                choice = input("\nEnter document number to add/remove (or 'd' when done, 'c' to cancel): ").strip()
                
                if choice.lower() == 'c':
                    return False
                elif choice.lower() == 'd':
                    if len(selected_indices) < 2:
                        print("Please select at least 2 documents for comparison.")
                        continue
                    else:
                        break
                        
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_documents):
                    if idx in selected_indices:
                        selected_indices.remove(idx)
                        print(f"Removed: {self.available_documents[idx]['name']}")
                    else:
                        selected_indices.append(idx)
                        print(f"Added: {self.available_documents[idx]['name']}")
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        self.selected_documents = [self.available_documents[idx]["name"] for idx in selected_indices]
        self.comparison_mode = True
        
        print(f"\nSelected documents for comparison: {', '.join(self.selected_documents)}")
        return True
    
    def select_model(self):
        """Let user select an Ollama model."""
        if not self.ollama_available or not self.available_models:
            print("Ollama is not available or no models are installed.")
            return
            
        print("\nAvailable Ollama models:")
        for i, model in enumerate(self.available_models):
            if model == self.current_model:
                print(f"  {i+1}. {model} (current)")
            else:
                print(f"  {i+1}. {model}")
        
        while True:
            try:
                choice = input("\nSelect model number (or Enter to keep current): ").strip()
                if not choice:
                    break
                    
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_models):
                    self.current_model = self.available_models[idx]
                    print(f"Selected model: {self.current_model}")
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def query_multiple_documents(self, query: str, document_names: List[str], limit: int = 5) -> Dict[str, Any]:
        """Query multiple documents and organize results by document."""
        all_results = {}
        
        for doc_name in document_names:
            # Query each document individually
            results = self.pipeline.query_knowledge_graph(query, document_name=doc_name, limit=limit)
            all_results[doc_name] = results
        
        return all_results
    
    def compare_documents(self, query: str, document_names: List[str]) -> Dict[str, Any]:
        """Compare information across multiple documents based on a query."""
        if not self.ollama_available:
            print("⚠️ Ollama is not available. Comparative analysis works best with LLM support.")
            return self.query_multiple_documents(query, document_names)
        
        # Get results from each document
        doc_results = self.query_multiple_documents(query, document_names)
        
        # Format content for comparison
        comparison_context = self._format_comparison_context(doc_results)
        
        # Create comparison prompt
        prompt = f"""You are a helpful assistant that compares information across multiple documents.

QUERY: {query}

DOCUMENT INFORMATION:
{comparison_context}

Please compare and contrast the information from these documents. Focus on:
1. Key similarities across documents
2. Important differences or unique information in each document
3. A balanced summary that highlights the most relevant comparative insights

Organize your answer with clear sections for similarities and differences. Always cite which document each piece of information comes from.
"""

        # Call Ollama for comparison
        try:
            import requests
            import json
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.current_model,
                    "prompt": prompt,
                    "temperature": 0.3,  # Lower temperature for more factual comparison
                    "max_tokens": 1500   # Longer response for comparison
                },
                timeout=120  # Longer timeout for comparison
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
                
                # Return both the comparison and the individual results
                return {
                    "comparison": full_response,
                    "individual_results": doc_results
                }
            else:
                error_msg = f"Error: Failed to get comparison from Ollama. Status code: {response.status_code}"
                return {
                    "comparison": error_msg,
                    "individual_results": doc_results
                }
        except Exception as e:
            error_msg = f"Error comparing documents: {str(e)}"
            return {
                "comparison": error_msg,
                "individual_results": doc_results
            }
    
    def _format_comparison_context(self, doc_results: Dict[str, Any]) -> str:
        """Format the document results for comparison context."""
        context = ""
        
        for doc_name, results in doc_results.items():
            context += f"\n--- DOCUMENT: {doc_name} ---\n\n"
            
            # Add chunks
            context += "RELEVANT TEXT CHUNKS:\n"
            for i, chunk in enumerate(results.get("chunks", [])):
                context += f"Chunk {i+1} (Score: {chunk.get('score', 0):.2f}):\n{chunk.get('text', '')}\n\n"
            
            # Add entities
            context += "KEY ENTITIES:\n"
            for entity in results.get("entities", []):
                context += f"- {entity.get('text', '')} (Type: {entity.get('type', '')}, Frequency: {entity.get('frequency', 0)})\n"
            
            # Add relations
            if results.get("relations"):
                context += "\nRELATIONSHIPS:\n"
                for relation in results.get("relations", []):
                    context += f"- {relation.get('from', '')} {relation.get('relation', '')} {relation.get('to', '')}\n"
            
            context += "\n" + "=" * 40 + "\n"
        
        return context
    
    def chat_loop(self):
        """Main chat loop for asking questions about the documents."""
        if not self.selected_documents:
            print("No documents selected.")
            return
        
        if self.comparison_mode:
            print(f"\n💬 Comparing documents: {', '.join(self.selected_documents)}")
            intro_text = "Ask comparative questions about these documents."
        else:
            print(f"\n💬 Chatting with document: {self.selected_documents[0]}")
            intro_text = "Ask questions about this document."
            
        print(intro_text)
        print("Type 'exit', 'quit', or 'q' to exit the chat.")
        print("Type 'switch' to select different document(s).")
        print("Type 'compare' to switch to comparison mode.")
        print("Type 'single' to switch to single document mode.")
        print("Type 'model' to change the Ollama model.")
        print("Type 'help' to see these commands again.")
        
        while True:
            question = input("\n📝 Your question: ").strip()
            
            if question.lower() in ('exit', 'quit', 'q'):
                break
            elif question.lower() == 'switch':
                if self.comparison_mode:
                    if self.select_documents_for_comparison():
                        print(f"\n💬 Now comparing documents: {', '.join(self.selected_documents)}")
                else:
                    if self.select_single_document():
                        print(f"\n💬 Now chatting with document: {self.selected_documents[0]}")
                continue
            elif question.lower() == 'compare':
                if self.select_documents_for_comparison():
                    print(f"\n💬 Comparing documents: {', '.join(self.selected_documents)}")
                continue
            elif question.lower() == 'single':
                if self.select_single_document():
                    print(f"\n💬 Chatting with document: {self.selected_documents[0]}")
                continue
            elif question.lower() == 'model':
                self.select_model()
                continue
            elif question.lower() == 'help':
                print("Commands:")
                print("  exit, quit, q - Exit the chat")
                print("  switch - Select different document(s)")
                print("  compare - Switch to comparison mode")
                print("  single - Switch to single document mode")
                print("  model - Change the Ollama model")
                print("  help - Show this help message")
                continue
            elif not question:
                continue
            
            print("\nSearching for information...")
            
            try:
                # Handle comparison mode
                if self.comparison_mode:
                    results = self.compare_documents(question, self.selected_documents)
                    
                    if "comparison" in results:
                        print("\n🔍 Document Comparison:")
                        print(results["comparison"])
                    else:
                        print("\n⚠️ No comparison available. Showing individual results:")
                        for doc_name, doc_results in results["individual_results"].items():
                            print(f"\n📄 Document: {doc_name}")
                            for i, chunk in enumerate(doc_results.get("chunks", [])[:2]):
                                print(f"Chunk {i+1} (Score: {chunk.get('score', 0):.2f}):")
                                print(f"{chunk.get('text', '')[:200]}...")
                
                # Handle single document mode
                else:
                    doc_name = self.selected_documents[0]
                    
                    if self.ollama_available:
                        # Use Ollama for complete RAG
                        try:
                            results = self.pipeline.chat_with_document(
                                question, 
                                document_name=doc_name,
                                model_name=self.current_model
                            )
                            
                            print("\n🤖 Answer:")
                            print(results["answer"])
                            
                            # Show supporting evidence
                            print("\n📊 Supporting evidence:")
                            for i, chunk in enumerate(results["chunks"][:2]):
                                print(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
                                print(f"{chunk['text'][:200]}...")
                        except Exception as e:
                            print(f"\n❌ Error generating answer: {str(e)}")
                            print("Falling back to retrieval only...")
                            
                            # Fall back to retrieval only
                            results = self.pipeline.query_knowledge_graph(
                                question, 
                                document_name=doc_name
                            )
                            
                            print("\n📚 Retrieved information:")
                            for i, chunk in enumerate(results["chunks"]):
                                print(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
                                print(f"{chunk['text'][:200]}...")
                    else:
                        # Just show retrieved information without LLM
                        results = self.pipeline.query_knowledge_graph(
                            question, 
                            document_name=doc_name
                        )
                        
                        print("\n📚 Retrieved information:")
                        for i, chunk in enumerate(results["chunks"]):
                            print(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
                            print(f"{chunk['text'][:200]}...")
                        
                        print("\n🔍 Key entities found:")
                        for i, entity in enumerate(results["entities"][:5]):
                            print(f"- {entity['text']} ({entity['type']})")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Try rephrasing your question or selecting a different document.")
    
    def run(self, uri, user, password):
        """Run the CLI application."""
        if not self.setup_connection(uri, user, password):
            return
        
        self.check_ollama()
        self.get_available_documents()
        
        if not self.available_documents:
            print("No documents found in the database. Please process PDFs first.")
            return
        
        # Ask the user if they want to compare documents or chat with a single document
        if len(self.available_documents) >= 2:
            print("\nDo you want to:")
            print("  1. Chat with a single document")
            print("  2. Compare multiple documents")
            
            while True:
                choice = input("\nEnter your choice (1/2): ").strip()
                if choice == '1':
                    if not self.select_single_document():
                        return
                    break
                elif choice == '2':
                    if not self.select_documents_for_comparison():
                        return
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            # Only one document available, select it automatically
            self.selected_documents = [self.available_documents[0]["name"]]
            self.comparison_mode = False
            print(f"Automatically selected the only available document: {self.selected_documents[0]}")
        
        self.chat_loop()
        
        # Clean up
        if self.pipeline:
            self.pipeline.close()
            print("Connection closed.")


def main():
    """Parse arguments and run the application."""
    parser = argparse.ArgumentParser(description="Chat with and compare PDF documents using Neo4j and Ollama")
    
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI (default: bolt://localhost:7687)")
    parser.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
    parser.add_argument("--password", help="Neo4j password (will prompt if not provided)")
    
    args = parser.parse_args()
    
    # Prompt for password if not provided
    password = args.password
    if not password:
        password = getpass("Enter Neo4j password: ")
    
    # Create and run the CLI
    cli = ComparativePDFChatCLI()
    cli.run(args.uri, args.user, password)


if __name__ == "__main__":
    main()