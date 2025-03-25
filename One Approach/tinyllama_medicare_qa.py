from neo4j import GraphDatabase
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import argparse

def query_neo4j(question, neo4j_uri, neo4j_user, neo4j_password):
    """Retrieve relevant sections from Neo4j based on the question"""
    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Extract keywords from question
    keywords = [word for word in question.lower().split() 
                if len(word) > 3 and word not in ['what', 'when', 'where', 'how', 'does', 'why', 'which', 'who', 'that', 'this', 'these', 'those']]
    
    # Simple query based on question type
    if any(word in question.lower() for word in ['eligibility', 'eligible', 'qualify']):
        query = """
        MATCH (s:Section)
        WHERE s.title CONTAINS 'eligibility' OR s.title CONTAINS 'requirement'
        RETURN s.title, LEFT(s.content, 300) as content
        LIMIT 2
        """
    elif any(word in question.lower() for word in ['pay', 'payment', 'cost', 'money', 'premium']):
        query = """
        MATCH (s:Section)
        WHERE s.title CONTAINS 'premium' OR s.title CONTAINS 'payment' OR s.title CONTAINS 'cost'
        RETURN s.title, LEFT(s.content, 300) as content
        LIMIT 2
        """
    elif any(word in question.lower() for word in ['move', 'moving', 'relocate', 'moved']):
        query = """
        MATCH (s:Section)
        WHERE s.content CONTAINS 'move' OR s.content CONTAINS 'moving'
        RETURN s.title, LEFT(s.content, 300) as content
        LIMIT 2
        """
    elif any(word in question.lower() for word in ['disenroll', 'disenrollment', 'leave', 'leaving', 'cancel']):
        query = """
        MATCH (s:Section)
        WHERE s.content CONTAINS 'disenroll' OR s.content CONTAINS 'leave the plan' 
        OR s.title CONTAINS 'ending' OR s.title CONTAINS 'cancel'
        RETURN s.title, LEFT(s.content, 300) as content
        LIMIT 2
        """
    elif len(keywords) > 0:
        # Use the first meaningful keyword
        keyword = keywords[0]
        query = f"""
        MATCH (s:Section)
        WHERE s.content CONTAINS '{keyword}'
        RETURN s.title, LEFT(s.content, 300) as content
        LIMIT 2
        """
    else:
        query = """
        MATCH (s:Section)
        RETURN s.title, LEFT(s.content, 300) as content
        LIMIT 1
        """
    
    # Execute query
    with driver.session() as session:
        results = list(session.run(query))
    
    driver.close()
    
    # Format results
    context = ""
    for record in results:
        title = record.get("s.title", "")
        content = record.get("content", "")
        context += f"{title}: {content}\n\n"
    
    return context

def load_tinyllama():
    """Load TinyLlama model with CPU-compatible settings"""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("Loading tokenizer...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model for CPU...")
    # Simpler loading without quantization for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Regular precision for CPU
        device_map="cpu"            # Force CPU
    )
    
    return model, tokenizer

def answer_with_tinyllama(question, context, model, tokenizer):
    """Generate answer using TinyLlama based on question and document context"""
    # Clean up memory before processing
    gc.collect()
    
    # Format prompt for TinyLlama in chat format
    prompt = f"""<human>: I have a question about a Medicare document. Here's some relevant information:

{context}

My question is: {question}</human>

<assistant>:"""
    
    # Ensure we don't exceed the model's context length (2048 tokens for TinyLlama)
    # If prompt is too long, truncate the context
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs.input_ids.shape[1] > 1024:  # Use half the context window as safety margin
        # Tokenize just the question and the format parts to find their length
        format_parts = f"""<human>: I have a question about a Medicare document. Here's some relevant information:

My question is: {question}</human>

<assistant>:"""
        format_tokens = tokenizer(format_parts, return_tensors="pt").input_ids.shape[1]
        
        # Calculate available space for context
        available_tokens = 1024 - format_tokens
        
        # Truncate context to fit
        context_parts = context.split("\n\n")
        shortened_context = ""
        for part in context_parts:
            temp_context = shortened_context + part + "\n\n"
            temp_prompt = f"""<human>: I have a question about a Medicare document. Here's some relevant information:

{temp_context}

My question is: {question}</human>

<assistant>:"""
            temp_tokens = tokenizer(temp_prompt, return_tensors="pt").input_ids.shape[1]
            
            if temp_tokens <= 1024:
                shortened_context = temp_context
            else:
                break
        
        # Recreate prompt with shortened context
        prompt = f"""<human>: I have a question about a Medicare document. Here's some relevant information:

{shortened_context}

My question is: {question}</human>

<assistant>:"""
    
    # Move inputs to the device the model is on (CPU in this case)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("Generating with TinyLlama (this might take a minute)...")
    # Generate response with conservative parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,         # Limit response length
            do_sample=True,             # Use sampling for more natural text
            temperature=0.7,            # Lower temperature for more focused answers
            top_p=0.9,                  # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return only the model's response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output.split("<assistant>:")[-1].strip()
    
    # Clean up memory after processing
    del outputs, inputs
    gc.collect()
    
    return response

def rule_based_answer(question, context):
    """Fallback function for rule-based answers"""
    if any(word in question.lower() for word in ['eligibility', 'eligible', 'qualify', 'requirement']):
        return "Based on the Medicare document, eligibility requirements typically include having both Medicare Part A and Part B, living in the service area, and being a United States citizen or lawfully present in the United States."
    
    elif any(word in question.lower() for word in ['pay', 'payment', 'cost', 'premium']):
        return "Based on the Medicare document, there are usually several payment options for premiums or penalties, including monthly billing, Electronic Funds Transfer (EFT), credit card payments, or deduction from Social Security checks."
    
    elif any(word in question.lower() for word in ['service area', 'location', 'where', 'area']) and not any(word in question.lower() for word in ['move', 'leave', 'moving']):
        return "The service area typically includes specific counties. For the AARP Medicare Advantage plan in the document, the service area is Los Angeles County in California."
    
    elif any(word in question.lower() for word in ['disenroll', 'disenrollment', 'leave', 'leaving', 'cancel']):
        return "When you disenroll from the plan, you typically have options including returning to Original Medicare or joining another Medicare Advantage plan. Special enrollment periods may apply when you move out of the service area."
    
    return f"Based on the Medicare document: {context}"

def medicare_qa_system(neo4j_uri, neo4j_user, neo4j_password):
    """Main function implementing the Medicare QA system"""
    print("\nTinyLlama Medicare Document Question-Answering System")
    print("Loading model (this may take a moment)...")
    
    # Load model once at startup
    try:
        model, tokenizer = load_tinyllama()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to rule-based answers.")
        model, tokenizer = None, None
        
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Ask a question about the Medicare document: ")
        if question.lower() == 'exit':
            break
        
        print("Retrieving information...")
        context = query_neo4j(question, neo4j_uri, neo4j_user, neo4j_password)
        
        if not context:
            print("\nAnswer:")
            print("I couldn't find relevant information about that in the Medicare document.")
            print("\n---")
            continue
        
        print("Generating answer...")
        
        # Try TinyLlama first and only fall back to rule-based if it fails
        if model is not None and tokenizer is not None:
            try:
                answer = answer_with_tinyllama(question, context, model, tokenizer)
            except Exception as e:
                print(f"Error using TinyLlama: {e}")
                print("Falling back to rule-based answer.")
                answer = rule_based_answer(question, context)
        else:
            answer = rule_based_answer(question, context)
        
        print("\nAnswer:")
        print(answer)
        print("\n---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyLlama Medicare QA System')
    parser.add_argument('--uri', required=True, help='Neo4j connection URI')
    parser.add_argument('--user', required=True, help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    
    args = parser.parse_args()
    
    medicare_qa_system(args.uri, args.user, args.password)