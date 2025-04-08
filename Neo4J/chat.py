# #!/usr/bin/env python3
# """
# Chat with PDF - A command-line tool to ask questions about PDF documents
# using a Neo4j knowledge graph and Ollama LLM.
# """

# import os
# import sys
# import argparse
# from getpass import getpass
# from neo4j import GraphDatabase

# # Import the GraphRAGPipeline or necessary components
# # You'll need to adjust the import based on your project structure
# try:
#     from pipeline import GraphRAGPipeline  # Import the main pipeline
# except ImportError:
#     print("Error: Could not import GraphRAGPipeline.")
#     print("Make sure the graphrag_pipeline.py file is in the same directory or in your PYTHONPATH.")
#     sys.exit(1)


# class PDFChatCLI:
#     """Command-line interface for chatting with PDF documents."""
    
#     def __init__(self):
#         self.pipeline = None
#         self.available_documents = []
#         self.current_document = None
#         self.ollama_available = False
#         self.available_models = []
#         self.current_model = "llama3"  # Default model
    
#     def setup_connection(self, uri, user, password):
#         """Connect to Neo4j and initialize the pipeline."""
#         try:
#             self.pipeline = GraphRAGPipeline(uri, user, password)
#             print("✅ Connected to Neo4j database.")
#             return True
#         except Exception as e:
#             print(f"❌ Error connecting to Neo4j: {str(e)}")
#             return False
    
#     def check_ollama(self):
#         """Check if Ollama is available and get available models."""
#         try:
#             import requests
#             response = requests.get("http://localhost:11434/api/tags", timeout=5)
#             if response.status_code == 200:
#                 self.ollama_available = True
#                 models = response.json().get("models", [])
#                 self.available_models = [model.get("name") for model in models]
#                 if self.available_models:
#                     self.current_model = self.available_models[0]  # Set first model as default
#                 print(f"✅ Ollama is available with {len(self.available_models)} models.")
#             else:
#                 print("❌ Ollama API returned an error.")
#                 self.ollama_available = False
#         except Exception as e:
#             print(f"❌ Ollama is not available: {str(e)}")
#             print("📝 Install Ollama from https://ollama.ai for better answers.")
#             self.ollama_available = False
    
#     def get_available_documents(self):
#         """Get list of available documents in the database."""
#         try:
#             with self.pipeline.driver.session() as session:
#                 result = session.run("""
#                     MATCH (d:Document)
#                     RETURN d.name AS name, count{(:Chunk)<-[:HAS_CHUNK]-(d)} AS chunks
#                     ORDER BY name
#                 """)
#                 self.available_documents = [(record["name"], record["chunks"]) for record in result]
                
#                 if not self.available_documents:
#                     print("No documents found in the database.")
#                 else:
#                     print(f"Found {len(self.available_documents)} documents:")
#                     for i, (name, chunks) in enumerate(self.available_documents):
#                         print(f"  {i+1}. {name} ({chunks} chunks)")
                
#                 # Set first document as default if available
#                 if self.available_documents:
#                     self.current_document = self.available_documents[0][0]
#         except Exception as e:
#             print(f"Error retrieving documents: {str(e)}")
    
#     def select_document(self):
#         """Let user select a document to chat with."""
#         if not self.available_documents:
#             print("No documents available. Please process a PDF first.")
#             return False
            
#         print("\nSelect a document to chat with:")
#         for i, (name, chunks) in enumerate(self.available_documents):
#             print(f"  {i+1}. {name} ({chunks} chunks)")
        
#         while True:
#             try:
#                 choice = input("\nEnter document number (or 'q' to quit): ").strip()
#                 if choice.lower() == 'q':
#                     return False
                    
#                 idx = int(choice) - 1
#                 if 0 <= idx < len(self.available_documents):
#                     self.current_document = self.available_documents[idx][0]
#                     print(f"Selected document: {self.current_document}")
#                     return True
#                 else:
#                     print("Invalid selection. Please try again.")
#             except ValueError:
#                 print("Please enter a valid number.")
    
#     def select_model(self):
#         """Let user select an Ollama model."""
#         if not self.ollama_available or not self.available_models:
#             print("Ollama is not available or no models are installed.")
#             return
            
#         print("\nAvailable Ollama models:")
#         for i, model in enumerate(self.available_models):
#             if model == self.current_model:
#                 print(f"  {i+1}. {model} (current)")
#             else:
#                 print(f"  {i+1}. {model}")
        
#         while True:
#             try:
#                 choice = input("\nSelect model number (or Enter to keep current): ").strip()
#                 if not choice:
#                     break
                    
#                 idx = int(choice) - 1
#                 if 0 <= idx < len(self.available_models):
#                     self.current_model = self.available_models[idx]
#                     print(f"Selected model: {self.current_model}")
#                     break
#                 else:
#                     print("Invalid selection. Please try again.")
#             except ValueError:
#                 print("Please enter a valid number.")
    
#     def chat_loop(self):
#         """Main chat loop for asking questions about the document."""
#         if not self.current_document:
#             print("No document selected.")
#             return
        
#         print(f"\n💬 Chatting with document: {self.current_document}")
#         print("Type 'exit', 'quit', or 'q' to exit the chat.")
#         print("Type 'switch' to select a different document.")
#         print("Type 'model' to change the Ollama model.")
#         print("Type 'help' to see these commands again.")
        
#         while True:
#             question = input("\n📝 Your question: ").strip()
            
#             if question.lower() in ('exit', 'quit', 'q'):
#                 break
#             elif question.lower() == 'switch':
#                 if self.select_document():
#                     print(f"\n💬 Now chatting with document: {self.current_document}")
#                 continue
#             elif question.lower() == 'model':
#                 self.select_model()
#                 continue
#             elif question.lower() == 'help':
#                 print("Commands:")
#                 print("  exit, quit, q - Exit the chat")
#                 print("  switch - Select a different document")
#                 print("  model - Change the Ollama model")
#                 print("  help - Show this help message")
#                 continue
#             elif not question:
#                 continue
            
#             print("\nSearching for information...")
            
#             if self.ollama_available:
#                 # Use Ollama for complete RAG
#                 results = self.pipeline.chat_with_document(
#                     question, 
#                     document_name=self.current_document,
#                     model_name=self.current_model
#                 )
                
#                 print("\n🤖 Answer:")
#                 print(results["answer"])
                
#                 # Show supporting evidence
#                 print("\n📊 Supporting evidence:")
#                 for i, chunk in enumerate(results["chunks"][:2]):
#                     print(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
#                     print(f"{chunk['text'][:200]}...")
#             else:
#                 # Just show retrieved information without LLM
#                 results = self.pipeline.query_knowledge_graph(
#                     question, 
#                     document_name=self.current_document
#                 )
                
#                 print("\n📚 Retrieved information:")
#                 for i, chunk in enumerate(results["chunks"]):
#                     print(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
#                     print(f"{chunk['text'][:200]}...")
                
#                 print("\n🔍 Key entities found:")
#                 for i, entity in enumerate(results["entities"][:5]):
#                     print(f"- {entity['text']} ({entity['type']})")
    
#     def run(self, uri, user, password):
#         """Run the CLI application."""
#         if not self.setup_connection(uri, user, password):
#             return
        
#         self.check_ollama()
#         self.get_available_documents()
        
#         if not self.available_documents:
#             print("No documents found in the database. Please process PDFs first.")
#             return
        
#         # If only one document is available, select it automatically
#         if len(self.available_documents) == 1:
#             self.current_document = self.available_documents[0][0]
#             print(f"Automatically selected the only available document: {self.current_document}")
#         else:
#             if not self.select_document():
#                 return
        
#         self.chat_loop()
        
#         # Clean up
#         if self.pipeline:
#             self.pipeline.close()
#             print("Connection closed.")


# def main():
#     """Parse arguments and run the application."""
#     parser = argparse.ArgumentParser(description="Chat with PDF documents using Neo4j and Ollama")
    
#     parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI (default: bolt://localhost:7687)")
#     parser.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
#     parser.add_argument("--password", help="Neo4j password (will prompt if not provided)")
    
#     args = parser.parse_args()
    
#     # Prompt for password if not provided
#     password = args.password
#     if not password:
#         password = getpass("Enter Neo4j password: ")
    
#     # Create and run the CLI
#     cli = PDFChatCLI()
#     cli.run(args.uri, args.user, password)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Chat with PDF - A command-line tool to ask questions about PDF documents
using a Neo4j knowledge graph and Ollama LLM.
"""

import os
import sys
import argparse
from getpass import getpass
from neo4j import GraphDatabase

# Import the GraphRAGPipeline
try:
    from pipeline import GraphRAGPipeline
except ImportError:
    print("Error: Could not import GraphRAGPipeline.")
    print("Make sure the pipeline.py file is in the same directory or in your PYTHONPATH.")
    sys.exit(1)


class PDFChatCLI:
    """Command-line interface for chatting with PDF documents."""
    
    def __init__(self):
        self.pipeline = None
        self.available_documents = []
        self.current_document = None
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
                    RETURN d.name AS name, count{(:Chunk)<-[:HAS_CHUNK]-(d)} AS chunks
                    ORDER BY name
                """)
                self.available_documents = [(record["name"], record["chunks"]) for record in result]
                
                if not self.available_documents:
                    print("No documents found in the database.")
                else:
                    print(f"Found {len(self.available_documents)} documents:")
                    for i, (name, chunks) in enumerate(self.available_documents):
                        print(f"  {i+1}. {name} ({chunks} chunks)")
                
                # Set first document as default if available
                if self.available_documents:
                    self.current_document = self.available_documents[0][0]
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            # Try alternative query for older Neo4j versions
            try:
                with self.pipeline.driver.session() as session:
                    result = session.run("""
                        MATCH (d:Document)
                        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                        RETURN d.name AS name, count(c) AS chunks
                        ORDER BY name
                    """)
                    self.available_documents = [(record["name"], record["chunks"]) for record in result]
                    
                    if not self.available_documents:
                        print("No documents found in the database.")
                    else:
                        print(f"Found {len(self.available_documents)} documents:")
                        for i, (name, chunks) in enumerate(self.available_documents):
                            print(f"  {i+1}. {name} ({chunks} chunks)")
                    
                    # Set first document as default if available
                    if self.available_documents:
                        self.current_document = self.available_documents[0][0]
            except Exception as e2:
                print(f"Error with alternative query: {str(e2)}")
    
    def select_document(self):
        """Let user select a document to chat with."""
        if not self.available_documents:
            print("No documents available. Please process a PDF first.")
            return False
            
        print("\nSelect a document to chat with:")
        for i, (name, chunks) in enumerate(self.available_documents):
            print(f"  {i+1}. {name} ({chunks} chunks)")
        
        while True:
            try:
                choice = input("\nEnter document number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return False
                    
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_documents):
                    self.current_document = self.available_documents[idx][0]
                    print(f"Selected document: {self.current_document}")
                    return True
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
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
    
    def chat_loop(self):
        """Main chat loop for asking questions about the document."""
        if not self.current_document:
            print("No document selected.")
            return
        
        print(f"\n💬 Chatting with document: {self.current_document}")
        print("Type 'exit', 'quit', or 'q' to exit the chat.")
        print("Type 'switch' to select a different document.")
        print("Type 'model' to change the Ollama model.")
        print("Type 'help' to see these commands again.")
        
        while True:
            question = input("\n📝 Your question: ").strip()
            
            if question.lower() in ('exit', 'quit', 'q'):
                break
            elif question.lower() == 'switch':
                if self.select_document():
                    print(f"\n💬 Now chatting with document: {self.current_document}")
                continue
            elif question.lower() == 'model':
                self.select_model()
                continue
            elif question.lower() == 'help':
                print("Commands:")
                print("  exit, quit, q - Exit the chat")
                print("  switch - Select a different document")
                print("  model - Change the Ollama model")
                print("  help - Show this help message")
                continue
            elif not question:
                continue
            
            print("\nSearching for information...")
            
            try:
                if self.ollama_available:
                    # Use Ollama for complete RAG
                    try:
                        results = self.pipeline.chat_with_document(
                            question, 
                            document_name=self.current_document,
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
                            document_name=self.current_document
                        )
                        
                        print("\n📚 Retrieved information:")
                        for i, chunk in enumerate(results["chunks"]):
                            print(f"Chunk {i+1} (Score: {chunk['score']:.2f}):")
                            print(f"{chunk['text'][:200]}...")
                else:
                    # Just show retrieved information without LLM
                    results = self.pipeline.query_knowledge_graph(
                        question, 
                        document_name=self.current_document
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
        
        # If only one document is available, select it automatically
        if len(self.available_documents) == 1:
            self.current_document = self.available_documents[0][0]
            print(f"Automatically selected the only available document: {self.current_document}")
        else:
            if not self.select_document():
                return
        
        self.chat_loop()
        
        # Clean up
        if self.pipeline:
            self.pipeline.close()
            print("Connection closed.")


def main():
    """Parse arguments and run the application."""
    parser = argparse.ArgumentParser(description="Chat with PDF documents using Neo4j and Ollama")
    
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI (default: bolt://localhost:7687)")
    parser.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
    parser.add_argument("--password", help="Neo4j password (will prompt if not provided)")
    
    args = parser.parse_args()
    
    # Prompt for password if not provided
    password = args.password
    if not password:
        password = getpass("Enter Neo4j password: ")
    
    # Create and run the CLI
    cli = PDFChatCLI()
    cli.run(args.uri, args.user, password)


if __name__ == "__main__":
    main()