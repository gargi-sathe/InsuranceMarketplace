# from neo4j import GraphDatabase
# import argparse
# import pandas as pd
# from tabulate import tabulate
# import re

# def list_documents(neo4j_uri, neo4j_user, neo4j_password):
#     """List all available documents in the database"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     with driver.session() as session:
#         result = session.run("MATCH (d:Document) RETURN d.title")
#         documents = [record["d.title"] for record in result]
    
#     driver.close()
#     return documents

# def extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, doc_title):
#     """Extract details from a specific Medicare plan with improved pattern matching"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     plan_details = {
#         "Plan": doc_title,
#         "Star Rating": "Not found",
#         "Health Deductible": "Not found",
#         "Drug Plan Deductible": "Not found",
#         "Maximum Out-of-Pocket": "Not found",
#         "Health Premium": "Not found",
#         "Drug Premium": "Not found",
#         "Part B Premium": "Not found",
#         "Features": {},
#         "Benefits": {},
#         "Drug Coverage": "Not found"
#     }
    
#     # Enhanced search patterns with more variations
#     search_patterns = {
#         "Star Rating": ["star rating", "plan rating", "overall rating", "star score", 
#                        "quality rating", "stars?\\s+rating", "rated\\s+\\d+(?:\\.\\d+)?\\s+stars?",
#                        "\\d+(?:\\.\\d+)?\\s+stars?", "\\d+(?:\\.\\d+)?\\s+star\\s+rating"],
                       
#         "Health Deductible": ["health deductible", "medical deductible", "plan deductible", 
#                              "in-network deductible", "deductible for medical", 
#                              "deductible for covered", "annual deductible", "yearly deductible"],
                             
#         "Drug Plan Deductible": ["drug deductible", "part d deductible", "prescription deductible",
#                                 "pharmacy deductible", "deductible for prescription", 
#                                 "deductible for part d", "part d annual deductible"],
                                
#         "Maximum Out-of-Pocket": ["maximum out-of-pocket", "out of pocket maximum", "max you pay",
#                                  "out-of-pocket limit", "maximum amount you pay", "moop", 
#                                  "annual out-of-pocket maximum", "yearly maximum"],
                                 
#         "Health Premium": ["health premium", "medical premium", "monthly premium", "plan premium",
#                           "premium you pay for", "premium for this plan"],
                          
#         "Drug Premium": ["drug premium", "part d premium", "prescription premium", 
#                         "premium for prescription", "premium for part d", 
#                         "additional premium for prescription"],
                        
#         "Part B Premium": ["part b premium", "medicare premium", "premium for part b",
#                           "medicare part b premium", "premium for medicare part b"]
#     }
    
#     # Enhanced feature patterns with more variations and negations
#     feature_patterns = {
#         "Vision": ["vision benefit", "vision coverage", "eye exam", "eyewear", "glasses", "vision care"],
#         "Dental": ["dental benefit", "dental coverage", "dental care", "dental service", "oral health"],
#         "Hearing": ["hearing benefit", "hearing aid", "hearing exam", "hearing coverage", "hearing care"],
#         "Transportation": ["transportation benefit", "rides to", "rides for", "medical transport", "non-emergency transport"],
#         "Fitness": ["fitness benefit", "gym membership", "exercise program", "silver sneakers", "fitness program"],
#         "Worldwide Emergency": ["worldwide emergency", "emergency outside", "international emergency", "emergency care abroad"],
#         "Over The Counter": ["over the counter", "otc benefit", "otc allowance", "otc items", "otc products"],
#         "In-Home Support": ["in-home support", "home health aide", "caregiver support", "home-based support"],
#         "Emergency Response": ["emergency response", "medical alert", "personal emergency", "pers device"],
#         "Bathroom Safety": ["bathroom safety", "safety device", "grab bars", "shower seat", "bathroom modification"],
#         "Meals": ["meal benefit", "meal delivery", "post-discharge meal", "prepared meals", "nutritional support"],
#         "Physical Exams": ["physical exam", "annual exam", "wellness visit", "routine physical", "preventive exam"],
#         "Telehealth": ["telehealth", "virtual visit", "remote access", "virtual care", "telemedicine"],
#         "Endodontics": ["endodontic", "root canal", "dental procedure", "tooth pulp", "dental service"],
#         "Periodontics": ["periodontic", "gum disease", "gum treatment", "periodontal", "dental service"]
#     }
    
#     # Enhanced benefit patterns
#     benefit_patterns = {
#         "Primary Doctor": ["primary doctor", "primary care", "pcp visit", "doctor visit"],
#         "Specialist": ["specialist", "specialist visit", "specialist doctor", "specialist care"],
#         "Diagnostic Tests": ["diagnostic test", "diagnostic procedure", "medical test", "diagnostic service"],
#         "Lab Services": ["lab service", "laboratory", "lab test", "lab work", "blood test"],
#         "Radiology": ["radiology", "imaging", "mri", "ct scan", "diagnostic imaging"],
#         "X-Rays": ["x-ray", "xray", "radiograph", "diagnostic x-ray"],
#         "Emergency Care": ["emergency care", "emergency room", "emergency service", "er visit"],
#         "Urgent Care": ["urgent care", "urgently needed", "urgent service"],
#         "Inpatient Hospital": ["inpatient hospital", "hospital stay", "inpatient care", "hospital admission"],
#         "Outpatient Hospital": ["outpatient hospital", "outpatient service", "outpatient care", "outpatient procedure"],
#         "Preventive Services": ["preventive service", "preventive care", "wellness service", "routine service"]
#     }
    
#     with driver.session() as session:
#         # Fetch main plan details with expanded approach
#         for detail, keywords in search_patterns.items():
#             # First try with exact match in title or content
#             query_parts = []
#             for keyword in keywords:
#                 query_parts.append(f"s.content CONTAINS '{keyword}'")
                
#             query = f"""
#             MATCH (s:Section {{document: $doc_title}})
#             WHERE {" OR ".join(query_parts)}
#             RETURN s.title, s.content
#             LIMIT 10
#             """
            
#             results = list(session.run(query, {"doc_title": doc_title}))
            
#             # Process results with improved pattern matching
#             if results:
#                 for result in results:
#                     content = result["s.content"].lower()
#                     title = result["s.title"].lower() if result["s.title"] else ""
                    
#                     if detail == "Star Rating":
#                         # Enhanced star rating detection
#                         matches = re.findall(r'(\d+(?:\.\d+)?)[ -]*stars?', content)
#                         if matches:
#                             plan_details[detail] = matches[0] + " stars"
#                             break
#                         # Try more patterns
#                         matches = re.findall(r'rated (\d+(?:\.\d+)?) stars?', content)
#                         if matches:
#                             plan_details[detail] = matches[0] + " stars"
#                             break
                            
#                     elif "Deductible" in detail:
#                         # Better deductible detection
#                         if "no deductible" in content or "0 deductible" in content or "$0 deductible" in content:
#                             plan_details[detail] = "$0"
#                             break
#                         # Find dollar amounts related to deductibles
#                         matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                         if matches:
#                             # Try to find nearby deductible context
#                             for match in matches:
#                                 surrounding_text = content.find('$' + match)
#                                 context_text = content[max(0, surrounding_text-30):min(len(content), surrounding_text+30)]
#                                 if any(keyword in context_text for keyword in keywords):
#                                     plan_details[detail] = "$" + match
#                                     break
#                             # If no contextualized match, take first match
#                             if plan_details[detail] == "Not found" and matches:
#                                 plan_details[detail] = "$" + matches[0]
#                             break
                                
#                     elif "Premium" in detail:
#                         # Better premium detection
#                         if "no premium" in content or "0 premium" in content or "$0 premium" in content:
#                             plan_details[detail] = "$0"
#                             break
#                         # Find dollar amounts with premium context
#                         matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                         if matches:
#                             # Try to find nearby premium context
#                             for match in matches:
#                                 surrounding_text = content.find('$' + match)
#                                 context_text = content[max(0, surrounding_text-30):min(len(content), surrounding_text+30)]
#                                 if any(keyword in context_text for keyword in keywords):
#                                     plan_details[detail] = "$" + match
#                                     break
#                             # If no contextualized match, take first match
#                             if plan_details[detail] == "Not found" and matches:
#                                 plan_details[detail] = "$" + matches[0]
#                             break
                            
#                     elif "Maximum" in detail:
#                         # Try identifying maximum out-of-pocket with context
#                         matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                         if matches:
#                             # Try to find nearby MOOP context
#                             for match in matches:
#                                 surrounding_text = content.find('$' + match)
#                                 context_text = content[max(0, surrounding_text-50):min(len(content), surrounding_text+50)]
#                                 if any(keyword in context_text for keyword in keywords):
#                                     plan_details[detail] = "$" + match
#                                     break
#                             # If no contextualized match, take first match
#                             if plan_details[detail] == "Not found" and matches:
#                                 plan_details[detail] = "$" + matches[0]
#                             break
        
#         # Fetch plan features with improved content analysis
#         for feature, patterns in feature_patterns.items():
#             # Build expanded pattern matching
#             query_parts = []
#             for pattern in patterns:
#                 query_parts.append(f"s.content CONTAINS '{pattern}'")
            
#             query = f"""
#             MATCH (s:Section {{document: $doc_title}})
#             WHERE {" OR ".join(query_parts)}
#             RETURN s.title, s.content
#             LIMIT 5
#             """
            
#             results = list(session.run(query, {"doc_title": doc_title}))
            
#             if results:
#                 # Analyze content more thoroughly
#                 combined_content = " ".join([result["s.content"].lower() for result in results])
                
#                 # Check for positive features
#                 positive_patterns = ["available", "covered", "included", "offered", "yes", 
#                                     "$0", "no copay", "free", "100%"]
#                 negative_patterns = ["not available", "not covered", "excluded", "not included",
#                                     "not offered", "no", "unavailable"]
                
#                 # First check for negative statements
#                 if any(neg_pattern in combined_content and any(feat_pattern in combined_content 
#                       for feat_pattern in patterns) for neg_pattern in negative_patterns):
#                     plan_details["Features"][feature] = "Not Available"
#                 # Then check for positive statements
#                 elif any(pos_pattern in combined_content and any(feat_pattern in combined_content 
#                         for feat_pattern in patterns) for pos_pattern in positive_patterns):
#                     plan_details["Features"][feature] = "Available"
#                 # Look for feature patterns themselves
#                 elif any(feat_pattern in combined_content for feat_pattern in patterns):
#                     # Default to available if feature is mentioned without negation
#                     plan_details["Features"][feature] = "Available"
#                 else:
#                     plan_details["Features"][feature] = "Unknown"
#             else:
#                 plan_details["Features"][feature] = "Not Found"
        
#         # Fetch benefit costs with improved analysis
#         for benefit, patterns in benefit_patterns.items():
#             query_parts = []
#             for pattern in patterns:
#                 query_parts.append(f"s.content CONTAINS '{pattern}'")
            
#             query = f"""
#             MATCH (s:Section {{document: $doc_title}})
#             WHERE {" OR ".join(query_parts)}
#             RETURN s.title, s.content
#             LIMIT 5
#             """
            
#             results = list(session.run(query, {"doc_title": doc_title}))
            
#             if results:
#                 # Analyze multiple results for cost information
#                 for result in results:
#                     content = result["s.content"].lower()
#                     title = result["s.title"].lower() if result["s.title"] else ""
                    
#                     # Look for dollar amounts near benefit mentions
#                     dollar_matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                     if dollar_matches:
#                         for pattern in patterns:
#                             if pattern in content:
#                                 for match in dollar_matches:
#                                     surrounding_text = content.find('$' + match)
#                                     context_text = content[max(0, surrounding_text-30):min(len(content), surrounding_text+30)]
#                                     if pattern in context_text:
#                                         plan_details["Benefits"][benefit] = "$" + match
#                                         break
#                                 if benefit in plan_details["Benefits"] and plan_details["Benefits"][benefit] != "Not Found":
#                                     break
                    
#                     # Check for zero-cost benefits
#                     for pattern in patterns:
#                         if pattern in content:
#                             if any(term in content for term in ["$0", "no copay", "free", "0 copay", "fully covered", "100% covered"]):
#                                 plan_details["Benefits"][benefit] = "$0"
#                                 break
                    
#                     # Check for coverage status
#                     for pattern in patterns:
#                         if pattern in content:
#                             if "covered" in content and "not covered" not in content:
#                                 if not plan_details["Benefits"].get(benefit) or plan_details["Benefits"][benefit] == "Not Found":
#                                     plan_details["Benefits"][benefit] = "Covered"
#                             elif "not covered" in content:
#                                 plan_details["Benefits"][benefit] = "Not Covered"
#                                 break
                    
#                     # If still nothing found, default to mentioning the benefit exists
#                     if not plan_details["Benefits"].get(benefit) or plan_details["Benefits"][benefit] == "Not Found":
#                         for pattern in patterns:
#                             if pattern in content:
#                                 plan_details["Benefits"][benefit] = "See Plan"
#                                 break
#             else:
#                 plan_details["Benefits"][benefit] = "Not Found"
        
#         # Enhanced drug coverage detection
#         drug_query = """
#         MATCH (s:Section {document: $doc_title})
#         WHERE s.content CONTAINS 'drug' OR s.content CONTAINS 'prescription' 
#               OR s.content CONTAINS 'medication' OR s.content CONTAINS 'pharmacy'
#               OR s.content CONTAINS 'formulary' OR s.title CONTAINS 'drug'
#               OR s.title CONTAINS 'prescription' OR s.title CONTAINS 'part d'
#         RETURN s.title, s.content
#         LIMIT 10
#         """
        
#         drug_results = list(session.run(drug_query, {"doc_title": doc_title}))
        
#         if drug_results:
#             combined_content = " ".join([result["s.content"].lower() for result in drug_results])
            
#             if "formulary" in combined_content:
#                 plan_details["Drug Coverage"] = "See Formulary"
#             elif "not covered" in combined_content and "drug" in combined_content:
#                 plan_details["Drug Coverage"] = "Not Covered"
#             elif any(term in combined_content for term in ["tier", "tiers", "drug list", "covered drugs"]):
#                 plan_details["Drug Coverage"] = "Covered (see details)"
#             elif "prescription" in combined_content or "drug" in combined_content:
#                 plan_details["Drug Coverage"] = "Covered"
    
#     driver.close()
#     return plan_details
        

# def search_all_document_content(neo4j_uri, neo4j_user, neo4j_password, doc_title, search_term):
#     """Search the entire document content for a term"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     query = """
#     MATCH (c:Chapter {document: $doc_title})-[:CONTAINS]->(s:Section)
#     WHERE toLower(s.content) CONTAINS toLower($search_term)
#     RETURN s.title, s.content
#     LIMIT 10
#     """
    
#     with driver.session() as session:
#         results = list(session.run(query, {"doc_title": doc_title, "search_term": search_term}))
    
#     driver.close()
#     return results

# def format_comparison(plan1_details, plan2_details):
#     """Format the comparison between two plans"""
#     comparison = {
#         "Criteria": ["Star Rating", "Health Deductible", "Drug Plan Deductible", 
#                      "Maximum Out-of-Pocket", "Health Premium", "Drug Premium", "Part B Premium"],
#         plan1_details["Plan"]: [plan1_details["Star Rating"], plan1_details["Health Deductible"],
#                                 plan1_details["Drug Plan Deductible"], plan1_details["Maximum Out-of-Pocket"],
#                                 plan1_details["Health Premium"], plan1_details["Drug Premium"], 
#                                 plan1_details["Part B Premium"]],
#         plan2_details["Plan"]: [plan2_details["Star Rating"], plan2_details["Health Deductible"],
#                                 plan2_details["Drug Plan Deductible"], plan2_details["Maximum Out-of-Pocket"],
#                                 plan2_details["Health Premium"], plan2_details["Drug Premium"],
#                                 plan2_details["Part B Premium"]]
#     }
    
#     # Create the main comparison DataFrame
#     df_main = pd.DataFrame(comparison)
    
#     # Features comparison
#     features = []
#     plan1_features = []
#     plan2_features = []
    
#     for feature in plan1_details["Features"]:
#         features.append(feature.title())
#         plan1_features.append(plan1_details["Features"][feature])
#         plan2_features.append(plan2_details["Features"].get(feature, "Not Found"))
    
#     df_features = pd.DataFrame({
#         "Feature": features,
#         plan1_details["Plan"]: plan1_features,
#         plan2_details["Plan"]: plan2_features
#     })
    
#     # Benefits comparison
#     benefits = []
#     plan1_benefits = []
#     plan2_benefits = []
    
#     for benefit in plan1_details["Benefits"]:
#         benefits.append(benefit.title())
#         plan1_benefits.append(plan1_details["Benefits"][benefit])
#         plan2_benefits.append(plan2_details["Benefits"].get(benefit, "Not Found"))
    
#     df_benefits = pd.DataFrame({
#         "Benefit": benefits,
#         plan1_details["Plan"]: plan1_benefits,
#         plan2_details["Plan"]: plan2_benefits
#     })
    
#     # Drug coverage comparison
#     df_drugs = pd.DataFrame({
#         "Coverage": ["Drug Coverage"],
#         plan1_details["Plan"]: [plan1_details["Drug Coverage"]],
#         plan2_details["Plan"]: [plan2_details["Drug Coverage"]]
#     })
    
#     return df_main, df_features, df_benefits, df_drugs

# def compare_medicare_plans(neo4j_uri, neo4j_user, neo4j_password):
#     """Main function for comparing Medicare plans"""
#     print("\nMedicare Plan Comparison Tool")
    
#     # List available documents
#     available_docs = list_documents(neo4j_uri, neo4j_user, neo4j_password)
#     if len(available_docs) < 2:
#         print("Need at least 2 documents for comparison. Please parse more Medicare plan documents.")
#         return
    
#     # Display documents
#     print("\nAvailable plans:")
#     for i, doc in enumerate(available_docs):
#         print(f"{i+1}. {doc}")
    
#     # Select two plans
#     try:
#         plan1_idx = int(input("\nSelect first plan (number): ")) - 1
#         plan2_idx = int(input("Select second plan (number): ")) - 1
        
#         if not (0 <= plan1_idx < len(available_docs) and 0 <= plan2_idx < len(available_docs)):
#             print("Invalid selection. Please try again.")
#             return
        
#         plan1_title = available_docs[plan1_idx]
#         plan2_title = available_docs[plan2_idx]
        
#         if plan1_title == plan2_title:
#             print("Please select two different plans.")
#             return
#     except ValueError:
#         print("Invalid input. Please enter a number.")
#         return
    
#     print(f"\nComparing: {plan1_title} vs {plan2_title}")
#     print("Extracting plan details (this may take a moment)...")
    
#     # Extract details for both plans
#     plan1_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan1_title)
#     plan2_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan2_title)
    
#     # Format the comparison
#     df_main, df_features, df_benefits, df_drugs = format_comparison(plan1_details, plan2_details)
    
#     # Display the comparison
#     print("\n== PLAN COMPARISON ==\n")
#     print("\nMain Plan Details:")
#     print(tabulate(df_main, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nFeatures:")
#     print(tabulate(df_features, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nBenefits and Costs:")
#     print(tabulate(df_benefits, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nDrug Coverage:")
#     print(tabulate(df_drugs, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nNote: 'Not found' indicates the information couldn't be extracted from the document.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Compare Medicare Plans')
#     parser.add_argument('--uri', required=True, help='Neo4j connection URI')
#     parser.add_argument('--user', required=True, help='Neo4j username')
#     parser.add_argument('--password', required=True, help='Neo4j password')
    
#     args = parser.parse_args()
    
#     compare_medicare_plans(args.uri, args.user, args.password)






# #####This code below does comparison decently
# from neo4j import GraphDatabase
# import argparse
# import pandas as pd
# from tabulate import tabulate
# import re
# import unicodedata

# def normalize_text(text):
#     """Normalize text to improve pattern matching"""
#     if not text:
#         return ""
#     # Convert to lowercase
#     text = text.lower()
#     # Normalize unicode characters
#     text = unicodedata.normalize('NFKD', text)
#     # Replace multiple spaces with single space
#     text = re.sub(r'\s+', ' ', text)
#     # Remove non-alphanumeric characters except spaces, periods, commas, and dollar signs
#     text = re.sub(r'[^\w\s.,\$]', ' ', text)
#     # Trim whitespace
#     text = text.strip()
#     return text

# def list_documents(neo4j_uri, neo4j_user, neo4j_password):
#     """List all available documents in the database"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     with driver.session() as session:
#         result = session.run("MATCH (d:Document) RETURN d.title")
#         documents = [record["d.title"] for record in result]
    
#     driver.close()
#     return documents

# def fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, search_term):
#     """Perform a fulltext search across all section content"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     with driver.session() as session:
#         query = """
#         CALL db.index.fulltext.queryNodes('sectionContentIndex', $search_term) 
#         YIELD node, score
#         WHERE node.document = $doc_title
#         RETURN node.title as title, node.content as content
#         ORDER BY score DESC
#         LIMIT 5
#         """
        
#         results = list(session.run(query, {"doc_title": doc_title, "search_term": search_term}))
    
#     driver.close()
#     return results

# def query_features_directly(neo4j_uri, neo4j_user, neo4j_password, doc_title):
#     """Query features directly from Feature nodes"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     with driver.session() as session:
#         query = """
#         MATCH (f:Feature {document: $doc_title})
#         RETURN f.name, f.available
#         """
        
#         results = list(session.run(query, {"doc_title": doc_title}))
    
#     driver.close()
    
#     features = {}
#     for record in results:
#         feature_name = record["f.name"]
#         available = record["f.available"]
        
#         # Convert feature names to display format
#         display_name = feature_name.replace("_", " ").title()
#         if feature_name == "otc":
#             display_name = "Over The Counter"
#         elif feature_name == "worldwide_emergency":
#             display_name = "Worldwide Emergency"
        
#         features[display_name] = "Available" if available else "Not Available"
    
#     return features

# def extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, doc_title):
#     """Extract details from a specific Medicare plan with improved extraction logic"""
#     driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
#     plan_details = {
#         "Plan": doc_title,
#         "Star Rating": "Not found",
#         "Health Deductible": "Not found",
#         "Drug Plan Deductible": "Not found",
#         "Maximum Out-of-Pocket": "Not found",
#         "Health Premium": "Not found",
#         "Drug Premium": "Not found",
#         "Part B Premium": "Not found",
#         "Features": {},
#         "Benefits": {},
#         "Drug Coverage": "Not found"
#     }
    
#     # First try to get features directly
#     plan_details["Features"] = query_features_directly(neo4j_uri, neo4j_user, neo4j_password, doc_title)
    
#     with driver.session() as session:
#         # First try to get metadata directly from document properties
#         doc_query = """
#         MATCH (d:Document {title: $doc_title})
#         RETURN properties(d) as props
#         """
        
#         doc_result = session.run(doc_query, {"doc_title": doc_title}).single()
#         if doc_result:
#             doc_props = doc_result["props"]
            
#             # Map document properties to plan details
#             if "star_rating" in doc_props:
#                 plan_details["Star Rating"] = doc_props["star_rating"] + " stars"
            
#             if "deductible" in doc_props:
#                 plan_details["Health Deductible"] = "$" + doc_props["deductible"]
            
#             if "drug_deductible" in doc_props:
#                 plan_details["Drug Plan Deductible"] = "$" + doc_props["drug_deductible"]
            
#             if "max_out_of_pocket" in doc_props:
#                 plan_details["Maximum Out-of-Pocket"] = "$" + doc_props["max_out_of_pocket"]
            
#             if "premium" in doc_props:
#                 plan_details["Health Premium"] = "$" + doc_props["premium"]
            
#             if "drug_premium" in doc_props:
#                 plan_details["Drug Premium"] = "$" + doc_props["drug_premium"]
            
#             if "part_b_premium" in doc_props:
#                 plan_details["Part B Premium"] = "$" + doc_props["part_b_premium"]
        
#         # Check for specific metadata on sections
#         section_query = """
#         MATCH (s:Section {document: $doc_title})
#         WHERE s.premium IS NOT NULL OR s.deductible IS NOT NULL 
#             OR s.max_out_of_pocket IS NOT NULL OR s.star_rating IS NOT NULL
#             OR s.drug_deductible IS NOT NULL OR s.drug_premium IS NOT NULL
#             OR s.part_b_premium IS NOT NULL
#         RETURN s.title, s.premium, s.deductible, s.max_out_of_pocket, s.star_rating,
#                s.drug_deductible, s.drug_premium, s.part_b_premium
#         """
        
#         section_results = list(session.run(section_query, {"doc_title": doc_title}))
#         for record in section_results:
#             if record["s.premium"] and plan_details["Health Premium"] == "Not found":
#                 plan_details["Health Premium"] = "$" + record["s.premium"]
            
#             if record["s.deductible"] and plan_details["Health Deductible"] == "Not found":
#                 # Try to determine if it's health or drug deductible based on title
#                 title = record["s.title"].lower() if record["s.title"] else ""
#                 if "drug" in title or "part d" in title or "prescription" in title:
#                     plan_details["Drug Plan Deductible"] = "$" + record["s.deductible"]
#                 else:
#                     plan_details["Health Deductible"] = "$" + record["s.deductible"]
            
#             if record["s.drug_deductible"] and plan_details["Drug Plan Deductible"] == "Not found":
#                 plan_details["Drug Plan Deductible"] = "$" + record["s.drug_deductible"]
            
#             if record["s.max_out_of_pocket"] and plan_details["Maximum Out-of-Pocket"] == "Not found":
#                 plan_details["Maximum Out-of-Pocket"] = "$" + record["s.max_out_of_pocket"]
            
#             if record["s.star_rating"] and plan_details["Star Rating"] == "Not found":
#                 plan_details["Star Rating"] = record["s.star_rating"] + " stars"
            
#             if record["s.drug_premium"] and plan_details["Drug Premium"] == "Not found":
#                 plan_details["Drug Premium"] = "$" + record["s.drug_premium"]
            
#             if record["s.part_b_premium"] and plan_details["Part B Premium"] == "Not found":
#                 plan_details["Part B Premium"] = "$" + record["s.part_b_premium"]
        
#         # Try to fetch benefit costs from Benefit nodes
#         benefit_query = """
#         MATCH (b:Benefit {document: $doc_title})
#         RETURN b.name, b.cost
#         """
        
#         benefit_results = list(session.run(benefit_query, {"doc_title": doc_title}))
        
#         benefit_mapping = {
#             "primary_care": "Primary Doctor",
#             "specialist": "Specialist",
#             "emergency": "Emergency Care",
#             "inpatient": "Inpatient Hospital",
#             "outpatient": "Outpatient Hospital",
#             "lab": "Lab Services",
#             "xray": "X-Rays",
#             "diagnostic": "Diagnostic Tests",
#             "urgent_care": "Urgent Care",
#             "preventive": "Preventive Services"
#         }
        
#         for record in benefit_results:
#             benefit_name = record["b.name"]
#             benefit_cost = record["b.cost"]
            
#             # Map benefit names to display names
#             display_name = benefit_mapping.get(benefit_name, benefit_name.replace("_", " ").title())
            
#             if benefit_cost == "0":
#                 plan_details["Benefits"][display_name] = "$0"
#             elif benefit_cost.isdigit():
#                 plan_details["Benefits"][display_name] = "$" + benefit_cost
#             else:
#                 plan_details["Benefits"][display_name] = benefit_cost.title()
        
#         # For missing details, use fulltext search as a fallback
#         if "Not found" in [plan_details[key] for key in ["Star Rating", "Health Deductible", "Drug Plan Deductible", "Maximum Out-of-Pocket", "Health Premium", "Drug Premium", "Part B Premium"]]:
#             # Star Rating search
#             if plan_details["Star Rating"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "star rating~0.7 OR rated stars~0.7")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     # Look for star ratings
#                     matches = re.findall(r'(\d+(?:\.\d+)?)[ -]*stars?', content)
#                     if matches:
#                         plan_details["Star Rating"] = matches[0] + " stars"
#                         break
            
#             # Health Deductible search
#             if plan_details["Health Deductible"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "deductible~0.8 OR plan deductible~0.8")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     # Filter out drug deductible mentions
#                     if "drug deductible" not in content and "part d deductible" not in content:
#                         if "no deductible" in content or "$0 deductible" in content:
#                             plan_details["Health Deductible"] = "$0"
#                             break
                        
#                         matches = re.findall(r'deductible.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                         if matches:
#                             plan_details["Health Deductible"] = "$" + matches[0]
#                             break
            
#             # Drug Plan Deductible search
#             if plan_details["Drug Plan Deductible"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "drug deductible~0.8 OR part d deductible~0.8")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     if "no deductible" in content or "$0 deductible" in content:
#                         plan_details["Drug Plan Deductible"] = "$0"
#                         break
                    
#                     matches = re.findall(r'deductible.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                     if matches:
#                         plan_details["Drug Plan Deductible"] = "$" + matches[0]
#                         break
            
#             # Maximum Out-of-Pocket search
#             if plan_details["Maximum Out-of-Pocket"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "maximum out of pocket~0.8 OR out-of-pocket~0.8 OR moop~")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     matches = re.findall(r'maximum out.?of.?pocket.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                     if matches:
#                         plan_details["Maximum Out-of-Pocket"] = "$" + matches[0]
#                         break
                    
#                     matches = re.findall(r'out.?of.?pocket.*?maximum.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                     if matches:
#                         plan_details["Maximum Out-of-Pocket"] = "$" + matches[0]
#                         break
            
#             # Health Premium search
#             if plan_details["Health Premium"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "premium~0.8 OR monthly premium~0.8")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     # Filter out drug premium mentions
#                     if "drug premium" not in content and "part d premium" not in content:
#                         if "no premium" in content or "$0 premium" in content:
#                             plan_details["Health Premium"] = "$0"
#                             break
                        
#                         matches = re.findall(r'premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                         if matches:
#                             plan_details["Health Premium"] = "$" + matches[0]
#                             break
            
#             # Drug Premium search
#             if plan_details["Drug Premium"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "drug premium~0.8 OR part d premium~0.8")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     if "no premium" in content or "$0 premium" in content:
#                         plan_details["Drug Premium"] = "$0"
#                         break
                    
#                     matches = re.findall(r'premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                     if matches:
#                         plan_details["Drug Premium"] = "$" + matches[0]
#                         break
            
#             # Part B Premium search
#             if plan_details["Part B Premium"] == "Not found":
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "part b premium~0.8 OR medicare premium~0.8")
#                 for result in results:
#                     content = normalize_text(result["content"])
                    
#                     matches = re.findall(r'part b premium.*?\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                     if matches:
#                         plan_details["Part B Premium"] = "$" + matches[0]
#                         break
        
#         # Search for missing features
#         feature_patterns = [
#             ("Vision", "vision~0.8 OR eye exam~0.8 OR eyewear~"),
#             ("Dental", "dental~0.8 OR oral health~0.8"),
#             ("Hearing", "hearing~0.8 OR hearing aid~0.8"),
#             ("Transportation", "transportation~0.8 OR ride~0.8"),
#             ("Fitness", "fitness~0.8 OR gym~0.8 OR exercise~0.8"),
#             ("Worldwide Emergency", "worldwide~0.8 OR international~0.8 AND emergency~"),
#             ("Over The Counter", "over the counter~0.8 OR otc~"),
#             ("Telehealth", "telehealth~0.8 OR virtual~0.8"),
#             ("Meals", "meal~0.8 OR food delivery~0.8"),
#             ("Physical Exams", "physical exam~0.8 OR annual exam~0.8"),
#             ("In-Home Support", "home support~0.8 OR in-home~0.8"),
#             ("Emergency Response", "emergency response~0.8 OR alert~0.8"),
#             ("Bathroom Safety", "bathroom~0.8 OR safety device~0.8"),
#             ("Endodontics", "endodontic~0.8 OR root canal~0.8"),
#             ("Periodontics", "periodontic~0.8 OR gum~0.8")
#         ]
        
#         for feature, search_term in feature_patterns:
#             if feature not in plan_details["Features"]:
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, search_term)
#                 if results:
#                     # Check if the feature is available or not
#                     combined_content = " ".join([normalize_text(result["content"]) for result in results])
                    
#                     if "not available" in combined_content or "not covered" in combined_content or "not included" in combined_content:
#                         plan_details["Features"][feature] = "Not Available"
#                     elif "available" in combined_content or "covered" in combined_content or "included" in combined_content:
#                         plan_details["Features"][feature] = "Available"
#                     else:
#                         # Default to available if mentioned
#                         plan_details["Features"][feature] = "Available"
        
#         # Search for missing benefits
#         benefit_patterns = [
#             ("Primary Doctor", "primary care~0.8 OR pcp~"),
#             ("Specialist", "specialist~0.8"),
#             ("Diagnostic Tests", "diagnostic~0.8 AND test~"),
#             ("Lab Services", "lab~0.8 OR laboratory~0.8"),
#             ("X-Rays", "x-ray~0.8 OR xray~"),
#             ("Emergency Care", "emergency~0.8 AND care~"),
#             ("Urgent Care", "urgent care~0.8"),
#             ("Inpatient Hospital", "inpatient~0.8 OR hospital stay~0.8"),
#             ("Outpatient Hospital", "outpatient~0.8"),
#             ("Preventive Services", "preventive~0.8 OR wellness~0.8")
#         ]
        
#         for benefit, search_term in benefit_patterns:
#             if benefit not in plan_details["Benefits"]:
#                 results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, search_term)
#                 if results:
#                     # Look for costs
#                     for result in results:
#                         content = normalize_text(result["content"])
                        
#                         if "$0" in content or "no copay" in content or "fully covered" in content:
#                             plan_details["Benefits"][benefit] = "$0"
#                             break
                        
#                         matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
#                         if matches:
#                             plan_details["Benefits"][benefit] = "$" + matches[0]
#                             break
                        
#                         if "covered" in content and "not covered" not in content:
#                             plan_details["Benefits"][benefit] = "Covered"
#                             break
#                         elif "not covered" in content:
#                             plan_details["Benefits"][benefit] = "Not Covered"
#                             break
        
#         # Drug coverage search
#         if plan_details["Drug Coverage"] == "Not found":
#             results = fulltext_search(neo4j_uri, neo4j_user, neo4j_password, doc_title, "drug coverage~0.8 OR prescription~0.8 OR formulary~")
#             if results:
#                 combined_content = " ".join([normalize_text(result["content"]) for result in results])
                
#                 if "formulary" in combined_content:
#                     plan_details["Drug Coverage"] = "See Formulary"
#                 elif "not covered" in combined_content:
#                     plan_details["Drug Coverage"] = "Not Covered"
#                 elif "drug" in combined_content or "prescription" in combined_content:
#                     plan_details["Drug Coverage"] = "Covered"
    
#     driver.close()
#     return plan_details

# def format_comparison(plan1_details, plan2_details):
#     """Format the comparison between two plans"""
#     comparison = {
#         "Criteria": ["Star Rating", "Health Deductible", "Drug Plan Deductible", 
#                      "Maximum Out-of-Pocket", "Health Premium", "Drug Premium", "Part B Premium"],
#         plan1_details["Plan"]: [plan1_details["Star Rating"], plan1_details["Health Deductible"],
#                                 plan1_details["Drug Plan Deductible"], plan1_details["Maximum Out-of-Pocket"],
#                                 plan1_details["Health Premium"], plan1_details["Drug Premium"], 
#                                 plan1_details["Part B Premium"]],
#         plan2_details["Plan"]: [plan2_details["Star Rating"], plan2_details["Health Deductible"],
#                                 plan2_details["Drug Plan Deductible"], plan2_details["Maximum Out-of-Pocket"],
#                                 plan2_details["Health Premium"], plan2_details["Drug Premium"],
#                                 plan2_details["Part B Premium"]]
#     }
    
#     # Create the main comparison DataFrame
#     df_main = pd.DataFrame(comparison)
    
#     # Features comparison
#     features = []
#     plan1_features = []
#     plan2_features = []
    
#     # All required feature names
#     required_features = [
#         "Vision", "Dental", "Hearing", "Transportation", "Fitness",
#         "Worldwide Emergency", "Over The Counter", "In-Home Support", 
#         "Emergency Response", "Bathroom Safety", "Meals", "Physical Exams",
#         "Telehealth", "Endodontics", "Periodontics"
#     ]
    
#     # First add all required features
#     for feature in required_features:
#         features.append(feature)
#         plan1_features.append(plan1_details["Features"].get(feature, "Not Found"))
#         plan2_features.append(plan2_details["Features"].get(feature, "Not Found"))
    
#     # Then add any additional features from either plan
#     all_features = set(list(plan1_details["Features"].keys()) + list(plan2_details["Features"].keys()))
#     for feature in sorted(all_features):
#         if feature not in required_features:
#             features.append(feature)
#             plan1_features.append(plan1_details["Features"].get(feature, "Not Found"))
#             plan2_features.append(plan2_details["Features"].get(feature, "Not Found"))
    
#     df_features = pd.DataFrame({
#         "Feature": features,
#         plan1_details["Plan"]: plan1_features,
#         plan2_details["Plan"]: plan2_features
#     })
    
#     # Benefits comparison
#     benefits = []
#     plan1_benefits = []
#     plan2_benefits = []
    
#     # Required benefit names
#     required_benefits = [
#         "Primary Doctor", "Specialist", "Diagnostic Tests", "Lab Services",
#         "Radiology", "X-Rays", "Emergency Care", "Urgent Care",
#         "Inpatient Hospital", "Outpatient Hospital", "Preventive Services"
#     ]
    
#     # First add all required benefits
#     for benefit in required_benefits:
#         benefits.append(benefit)
#         plan1_benefits.append(plan1_details["Benefits"].get(benefit, "Not Found"))
#         plan2_benefits.append(plan2_details["Benefits"].get(benefit, "Not Found"))
    
#     # Then add any additional benefits from either plan
#     all_benefits = set(list(plan1_details["Benefits"].keys()) + list(plan2_details["Benefits"].keys()))
#     for benefit in sorted(all_benefits):
#         if benefit not in required_benefits:
#             benefits.append(benefit)
#             plan1_benefits.append(plan1_details["Benefits"].get(benefit, "Not Found"))
#             plan2_benefits.append(plan2_details["Benefits"].get(benefit, "Not Found"))
    
#     df_benefits = pd.DataFrame({
#         "Benefit": benefits,
#         plan1_details["Plan"]: plan1_benefits,
#         plan2_details["Plan"]: plan2_benefits
#     })
    
#     # Drug coverage comparison
#     df_drugs = pd.DataFrame({
#         "Coverage": ["Drug Coverage"],
#         plan1_details["Plan"]: [plan1_details["Drug Coverage"]],
#         plan2_details["Plan"]: [plan2_details["Drug Coverage"]]
#     })
    
#     return df_main, df_features, df_benefits, df_drugs

# def compare_medicare_plans(neo4j_uri, neo4j_user, neo4j_password):
#     """Main function for comparing Medicare plans"""
#     print("\nMedicare Plan Comparison Tool")
    
#     # List available documents
#     available_docs = list_documents(neo4j_uri, neo4j_user, neo4j_password)
#     if len(available_docs) < 2:
#         print("Need at least 2 documents for comparison. Please parse more Medicare plan documents.")
#         return
    
#     # Display documents
#     print("\nAvailable plans:")
#     for i, doc in enumerate(available_docs):
#         print(f"{i+1}. {doc}")
    
#     # Select two plans
#     try:
#         plan1_idx = int(input("\nSelect first plan (number): ")) - 1
#         plan2_idx = int(input("Select second plan (number): ")) - 1
        
#         if not (0 <= plan1_idx < len(available_docs) and 0 <= plan2_idx < len(available_docs)):
#             print("Invalid selection. Please try again.")
#             return
        
#         plan1_title = available_docs[plan1_idx]
#         plan2_title = available_docs[plan2_idx]
        
#         if plan1_title == plan2_title:
#             print("Please select two different plans.")
#             return
#     except ValueError:
#         print("Invalid input. Please enter a number.")
#         return
    
#     print(f"\nComparing: {plan1_title} vs {plan2_title}")
#     print("Extracting plan details (this may take a moment)...")
    
#     # Extract details for both plans
#     plan1_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan1_title)
#     plan2_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan2_title)
    
#     # Format the comparison
#     df_main, df_features, df_benefits, df_drugs = format_comparison(plan1_details, plan2_details)
    
#     # Display the comparison
#     print("\n== PLAN COMPARISON ==\n")
#     print("\nMain Plan Details:")
#     print(tabulate(df_main, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nFeatures:")
#     print(tabulate(df_features, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nBenefits and Costs:")
#     print(tabulate(df_benefits, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nDrug Coverage:")
#     print(tabulate(df_drugs, headers='keys', tablefmt='grid', showindex=False))
    
#     print("\nNote: 'Not found' indicates the information couldn't be extracted from the document.")
    
#     # Ask if user would like to enter QA mode
#     qa_choice = input("\nWould you like to ask specific questions about these plans? (y/n): ")
#     if qa_choice.lower() == 'y':
#         interactive_qa_mode(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title)
# def extract_medical_conditions(question):
#     """Extract medical conditions from user question"""
#     # Dictionary of common medical conditions to look for
#     conditions = [
#         "diabetes", "heart disease", "asthma", "cancer", "arthritis", 
#         "hypertension", "high blood pressure", "depression", "anxiety",
#         "copd", "alzheimer", "dementia", "kidney disease", "stroke",
#         "multiple sclerosis", "parkinson", "chronic pain"
#     ]
    
#     found_conditions = []
#     normalized_question = question.lower()
    
#     for condition in conditions:
#         if condition in normalized_question:
#             found_conditions.append(condition)
    
#     # If no specific conditions found, extract potential health-related terms
#     if not found_conditions:
#         health_indicators = ["medicine", "prescription", "drug", "therapy", "specialist", 
#                            "condition", "doctor", "hospital", "surgery", "medical"]
        
#         for indicator in health_indicators:
#             if indicator in normalized_question:
#                 found_conditions.append(indicator)
#                 break
    
#     # Default to general health if no conditions found
#     if not found_conditions:
#         found_conditions = ["health care"]
    
#     return found_conditions

# def get_condition_related_services(conditions):
#     """Get services related to specific medical conditions"""
#     condition_services = {
#         "diabetes": ["specialist", "endocrinologist", "lab", "drug", "supplies", "preventive"],
#         "heart disease": ["cardiologist", "specialist", "diagnostic", "emergency", "hospital"],
#         "asthma": ["specialist", "pulmonologist", "prescription", "emergency"],
#         "cancer": ["oncologist", "specialist", "hospital", "radiation", "chemotherapy"],
#         "arthritis": ["rheumatologist", "specialist", "therapy", "pain management"],
#         "hypertension": ["specialist", "lab", "diagnostic", "preventive"],
#         "depression": ["mental health", "therapy", "psychiatric", "prescription"],
#         "anxiety": ["mental health", "therapy", "psychiatric", "prescription"],
#         "chronic pain": ["pain management", "specialist", "therapy", "prescription"],
#         "health care": ["primary care", "specialist", "preventive", "hospital", "emergency"]
#     }
    
#     services = []
#     for condition in conditions:
#         condition_key = next((k for k in condition_services.keys() if k in condition), "health care")
#         services.extend(condition_services[condition_key])
    
#     return list(set(services))  # Remove duplicates

# def compare_and_recommend(results, plan1_title, plan2_title, conditions):
#     """Compare results and generate a recommendation based on plan coverage for conditions"""
#     plan1_data = results[plan1_title]
#     plan2_data = results[plan2_title]
    
#     comparison_points = []
#     plan1_score = 0
#     plan2_score = 0
    
#     # Compare direct condition mentions
#     for condition in conditions:
#         if condition in plan1_data and condition in plan2_data:
#             # Both plans mention the condition
#             plan1_mentions = len(plan1_data[condition])
#             plan2_mentions = len(plan2_data[condition])
            
#             if plan1_mentions > plan2_mentions:
#                 comparison_points.append(f"{plan1_title} has more detailed information about {condition} care.")
#                 plan1_score += 1
#             elif plan2_mentions > plan1_mentions:
#                 comparison_points.append(f"{plan2_title} has more detailed information about {condition} care.")
#                 plan2_score += 1
#         elif condition in plan1_data:
#             comparison_points.append(f"Only {plan1_title} specifically mentions {condition} care.")
#             plan1_score += 2
#         elif condition in plan2_data:
#             comparison_points.append(f"Only {plan2_title} specifically mentions {condition} care.")
#             plan2_score += 2
    
#     # Compare service costs
#     service_keys = [k for k in plan1_data.keys() if k.endswith('_services')] + [k for k in plan2_data.keys() if k.endswith('_services')]
#     service_keys = list(set(service_keys))  # Remove duplicates
    
#     for service_key in service_keys:
#         service_name = service_key.replace('_services', '')
        
#         plan1_service = plan1_data.get(service_key, [])
#         plan2_service = plan2_data.get(service_key, [])
        
#         if plan1_service and plan2_service:
#             # Compare costs if available
#             plan1_cost = extract_cost(plan1_service[0].get('cost', 'not found'))
#             plan2_cost = extract_cost(plan2_service[0].get('cost', 'not found'))
            
#             if plan1_cost is not None and plan2_cost is not None:
#                 if plan1_cost < plan2_cost:
#                     comparison_points.append(f"{plan1_title} has lower costs for {service_name} services (${plan1_cost} vs ${plan2_cost}).")
#                     plan1_score += 2
#                 elif plan2_cost < plan1_cost:
#                     comparison_points.append(f"{plan2_title} has lower costs for {service_name} services (${plan2_cost} vs ${plan1_cost}).")
#                     plan2_score += 2
#             elif plan1_cost is not None:
#                 comparison_points.append(f"{plan1_title} has defined costs for {service_name} services (${plan1_cost}).")
#                 plan1_score += 1
#             elif plan2_cost is not None:
#                 comparison_points.append(f"{plan2_title} has defined costs for {service_name} services (${plan2_cost}).")
#                 plan2_score += 1
#         elif plan1_service:
#             comparison_points.append(f"{plan1_title} specifically covers {service_name} services.")
#             plan1_score += 1
#         elif plan2_service:
#             comparison_points.append(f"{plan2_title} specifically covers {service_name} services.")
#             plan2_score += 1
    
#     # Generate recommendation
#     recommendation = "Based on the available information:\n\n"
    
#     for point in comparison_points[:5]:  # Limit to top 5 points
#         recommendation += f"- {point}\n"
    
#     if plan1_score > plan2_score:
#         recommendation += f"\nOverall, {plan1_title} appears to be better suited for your needs with {conditions[0]}."
#     elif plan2_score > plan1_score:
#         recommendation += f"\nOverall, {plan2_title} appears to be better suited for your needs with {conditions[0]}."
#     else:
#         recommendation += f"\nBoth plans appear similar in their coverage for {conditions[0]}. Consider other factors like provider networks."
    
#     return recommendation

# def extract_cost(cost_str):
#     """Extract numeric cost from cost string"""
#     if cost_str == "0" or cost_str == "covered":
#         return 0
    
#     try:
#         if cost_str.isdigit():
#             return int(cost_str)
        
#         if cost_str.startswith("$"):
#             cost_value = cost_str.replace("$", "").replace(",", "")
#             return float(cost_value)
#     except:
#         pass
    
#     return None

# def interactive_qa_mode(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title):
#     """Interactive QA mode for comparing plans for specific needs"""
#     print("\nPlan Comparison Q&A Mode")
#     print("Ask questions like 'Which plan is better for diabetes?' or 'If I need regular specialist visits, which plan should I choose?'")
#     print("Type 'exit' to return to main menu\n")
    
#     while True:
#         question = input("Your question: ")
#         if question.lower() == 'exit':
#             break
        
#         print("Analyzing plans...")
#         recommendation = ask_plan_question(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title, question)
        
#         print("\nRecommendation:")
#         print(recommendation)
#         print("\n---")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Compare Medicare Plans')
#     parser.add_argument('--uri', required=True, help='Neo4j connection URI')
#     parser.add_argument('--user', required=True, help='Neo4j username')
#     parser.add_argument('--password', required=True, help='Neo4j password')
    
#     args = parser.parse_args()
    
#     compare_medicare_plans(args.uri, args.user, args.password)




from neo4j import GraphDatabase
import argparse
import pandas as pd
from tabulate import tabulate
import re
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

def list_documents(neo4j_uri, neo4j_user, neo4j_password):
    """List all available documents in the database"""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN d.title")
        documents = [record["d.title"] for record in result]
    
    driver.close()
    return documents

def search_section_content(neo4j_uri, neo4j_user, neo4j_password, doc_title, search_term):
    """Search for content in document sections"""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session() as session:
        query = """
        MATCH (s:Section {document: $doc_title})
        WHERE toLower(s.content) CONTAINS toLower($search_term)
        RETURN s.title as title, s.content as content
        LIMIT 5
        """
        
        results = list(session.run(query, {"doc_title": doc_title, "search_term": search_term}))
    
    driver.close()
    return results

def detect_features_from_content(neo4j_uri, neo4j_user, neo4j_password, doc_title):
    """Detect features by directly searching section content"""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    features = {}
    feature_patterns = {
        "Vision": ["vision", "eye exam", "eyewear", "glasses"],
        "Dental": ["dental", "dentist", "oral health"],
        "Hearing": ["hearing", "hearing aid", "audiologist"],
        "Transportation": ["transportation", "ride", "transit"],
        "Fitness": ["fitness", "gym", "exercise", "silver sneakers"],
        "Worldwide Emergency": ["worldwide emergency", "international emergency"],
        "Over The Counter": ["over the counter", "otc", "otc benefit"],
        "In-Home Support": ["in-home support", "home health", "caregiver"],
        "Emergency Response": ["emergency response", "medical alert", "alert system"],
        "Bathroom Safety": ["bathroom safety", "safety device", "grab bars"],
        "Meals": ["meal", "meal delivery", "food delivery"],
        "Physical Exams": ["physical exam", "annual exam", "wellness visit"],
        "Telehealth": ["telehealth", "virtual visit", "telemedicine"],
        "Endodontics": ["endodontic", "root canal"],
        "Periodontics": ["periodontic", "gum disease", "gum treatment"]
    }
    
    with driver.session() as session:
        for feature_name, keywords in feature_patterns.items():
            found = False
            # Search sections for feature keywords
            for keyword in keywords:
                query = """
                MATCH (s:Section {document: $doc_title})
                WHERE toLower(s.content) CONTAINS toLower($keyword)
                RETURN s.content LIMIT 1
                """
                result = session.run(query, {"doc_title": doc_title, "keyword": keyword})
                record = result.single()
                
                if record:
                    content = record["s.content"].lower()
                    # Check if feature is available
                    if "not covered" in content or "not available" in content or "not included" in content:
                        features[feature_name] = "Not Available"
                    else:
                        features[feature_name] = "Available"
                    found = True
                    break
            
            # If feature not found in any sections
            if not found:
                features[feature_name] = "Not Found"
    
    driver.close()
    return features

def extract_benefit_info(neo4j_uri, neo4j_user, neo4j_password, doc_title):
    """Extract benefit information by searching section content"""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    benefits = {}
    benefit_patterns = {
        "Primary Doctor": ["primary care", "pcp", "doctor visit"],
        "Specialist": ["specialist", "specialist visit"],
        "Diagnostic Tests": ["diagnostic test", "diagnostic procedure"],
        "Lab Services": ["lab service", "laboratory", "lab test"],
        "X-Rays": ["x-ray", "xray", "radiograph"],
        "Emergency Care": ["emergency care", "emergency room", "er visit"],
        "Urgent Care": ["urgent care", "urgently needed care"],
        "Inpatient Hospital": ["inpatient hospital", "hospital stay"],
        "Outpatient Hospital": ["outpatient hospital", "outpatient service"],
        "Preventive Services": ["preventive service", "preventive care"]
    }
    
    with driver.session() as session:
        for benefit_name, keywords in benefit_patterns.items():
            for keyword in keywords:
                query = """
                MATCH (s:Section {document: $doc_title})
                WHERE toLower(s.content) CONTAINS toLower($keyword)
                RETURN s.content LIMIT 3
                """
                results = list(session.run(query, {"doc_title": doc_title, "keyword": keyword}))
                
                if results:
                    # Look for cost information in the content
                    for result in results:
                        content = result["s.content"].lower()
                        
                        # Check for $0 or no copay
                        if "$0" in content or "no copay" in content or "fully covered" in content:
                            benefits[benefit_name] = "$0"
                            break
                        
                        # Look for dollar amounts
                        matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
                        if matches:
                            # Try to ensure the dollar amount is related to the benefit
                            for match in matches:
                                surrounding_text = content.find('$' + match)
                                context = content[max(0, surrounding_text-50):min(len(content), surrounding_text+50)]
                                if any(kw in context for kw in keywords):
                                    benefits[benefit_name] = "$" + match
                                    break
                            
                            # If no specific match found, use the first dollar amount
                            if benefit_name not in benefits:
                                benefits[benefit_name] = "$" + matches[0]
                            break
                        
                        # Check if simply covered
                        if "covered" in content and "not covered" not in content:
                            benefits[benefit_name] = "Covered"
                            break
                        elif "not covered" in content:
                            benefits[benefit_name] = "Not Covered"
                            break
                
                # If we found information, move to next benefit
                if benefit_name in benefits:
                    break
            
            # If no information found after checking all keywords
            if benefit_name not in benefits:
                benefits[benefit_name] = "Not Found"
    
    driver.close()
    return benefits

def extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, doc_title):
    """Extract details from a specific Medicare plan by searching sections"""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    plan_details = {
        "Plan": doc_title,
        "Star Rating": "Not found",
        "Health Deductible": "Not found",
        "Drug Plan Deductible": "Not found",
        "Maximum Out-of-Pocket": "Not found",
        "Health Premium": "Not found",
        "Drug Premium": "Not found",
        "Part B Premium": "Not found",
        "Features": {},
        "Benefits": {},
        "Drug Coverage": "Not found"
    }
    
    # Get features by searching section content
    plan_details["Features"] = detect_features_from_content(neo4j_uri, neo4j_user, neo4j_password, doc_title)
    
    # Get benefits by searching section content
    plan_details["Benefits"] = extract_benefit_info(neo4j_uri, neo4j_user, neo4j_password, doc_title)
    
    with driver.session() as session:
        # First try to get metadata directly from document properties
        doc_query = """
        MATCH (d:Document {title: $doc_title})
        RETURN properties(d) as props
        """
        
        doc_result = session.run(doc_query, {"doc_title": doc_title}).single()
        if doc_result:
            doc_props = doc_result["props"]
            
            # Map document properties to plan details if they exist
            if "star_rating" in doc_props:
                plan_details["Star Rating"] = doc_props["star_rating"] + " stars"
            
            if "deductible" in doc_props:
                plan_details["Health Deductible"] = "$" + doc_props["deductible"]
            
            if "drug_deductible" in doc_props:
                plan_details["Drug Plan Deductible"] = "$" + doc_props["drug_deductible"]
            
            if "max_out_of_pocket" in doc_props:
                plan_details["Maximum Out-of-Pocket"] = "$" + doc_props["max_out_of_pocket"]
            
            if "premium" in doc_props:
                plan_details["Health Premium"] = "$" + doc_props["premium"]
            
            if "drug_premium" in doc_props:
                plan_details["Drug Premium"] = "$" + doc_props["drug_premium"]
            
            if "part_b_premium" in doc_props:
                plan_details["Part B Premium"] = "$" + doc_props["part_b_premium"]
        
        # For any missing details, search section content directly
        search_terms = {
            "Star Rating": ["star rating", "rated stars", "plan rating"],
            "Health Deductible": ["deductible", "health deductible", "medical deductible"],
            "Drug Plan Deductible": ["drug deductible", "part d deductible", "prescription deductible"],
            "Maximum Out-of-Pocket": ["maximum out of pocket", "out-of-pocket maximum", "moop"],
            "Health Premium": ["premium", "monthly premium", "plan premium"],
            "Drug Premium": ["drug premium", "part d premium", "prescription premium"],
            "Part B Premium": ["part b premium", "medicare premium"],
            "Drug Coverage": ["drug coverage", "prescription coverage", "formulary"]
        }
        
        for detail, terms in search_terms.items():
            if plan_details[detail] == "Not found":
                for term in terms:
                    results = search_section_content(neo4j_uri, neo4j_user, neo4j_password, doc_title, term)
                    
                    for result in results:
                        content = normalize_text(result["content"])
                        
                        # Star Rating
                        if detail == "Star Rating":
                            matches = re.findall(r'(\d+(?:\.\d+)?)[ -]*stars?', content)
                            if matches:
                                plan_details[detail] = matches[0] + " stars"
                                break
                        
                        # Deductibles
                        elif "Deductible" in detail:
                            if "no deductible" in content or "$0" in content:
                                plan_details[detail] = "$0"
                                break
                            
                            # For drug deductible, make sure we're looking at drug content
                            if detail == "Drug Plan Deductible" and not any(drug_term in content for drug_term in ["drug", "part d", "prescription"]):
                                continue
                            
                            # For health deductible, avoid drug content
                            if detail == "Health Deductible" and any(drug_term in content for drug_term in ["drug deductible", "part d deductible"]):
                                continue
                            
                            matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
                            if matches:
                                for match in matches:
                                    surrounding_text = content.find('$' + match)
                                    context = content[max(0, surrounding_text-50):min(len(content), surrounding_text+50)]
                                    if "deductible" in context:
                                        plan_details[detail] = "$" + match
                                        break
                                
                                # If no specific match found, use the first dollar amount
                                if plan_details[detail] == "Not found":
                                    plan_details[detail] = "$" + matches[0]
                                break
                        
                        # Maximum Out-of-Pocket
                        elif detail == "Maximum Out-of-Pocket":
                            if "$0" in content and "out of pocket" in content:
                                plan_details[detail] = "$0"
                                break
                            
                            matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
                            if matches:
                                for match in matches:
                                    surrounding_text = content.find('$' + match)
                                    context = content[max(0, surrounding_text-50):min(len(content), surrounding_text+50)]
                                    if any(max_term in context for max_term in ["maximum out", "out of pocket"]):
                                        plan_details[detail] = "$" + match
                                        break
                                
                                # If no specific match found, use the first dollar amount
                                if plan_details[detail] == "Not found":
                                    plan_details[detail] = "$" + matches[0]
                                break
                        
                        # Premiums
                        elif "Premium" in detail:
                            if "no premium" in content or "$0 premium" in content:
                                plan_details[detail] = "$0"
                                break
                            
                            # For drug premium, ensure it's related to drugs
                            if detail == "Drug Premium" and not any(drug_term in content for drug_term in ["drug", "part d", "prescription"]):
                                continue
                            
                            # For health premium, avoid drug content
                            if detail == "Health Premium" and any(drug_term in content for drug_term in ["drug premium", "part d premium"]):
                                continue
                            
                            matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content)
                            if matches:
                                for match in matches:
                                    surrounding_text = content.find('$' + match)
                                    context = content[max(0, surrounding_text-50):min(len(content), surrounding_text+50)]
                                    if "premium" in context:
                                        plan_details[detail] = "$" + match
                                        break
                                
                                # If no specific match found, use the first dollar amount
                                if plan_details[detail] == "Not found":
                                    plan_details[detail] = "$" + matches[0]
                                break
                        
                        # Drug Coverage
                        elif detail == "Drug Coverage":
                            if "not covered" in content and any(drug_term in content for drug_term in ["drug", "prescription"]):
                                plan_details[detail] = "Not Covered"
                                break
                            elif "formulary" in content:
                                plan_details[detail] = "See Formulary"
                                break
                            elif any(drug_term in content for drug_term in ["drug coverage", "prescription coverage"]):
                                plan_details[detail] = "Covered"
                                break
                    
                    # If we found the detail, break out of the terms loop
                    if plan_details[detail] != "Not found":
                        break
    
    driver.close()
    return plan_details

def format_comparison(plan1_details, plan2_details):
    """Format the comparison between two plans"""
    comparison = {
        "Criteria": ["Star Rating", "Health Deductible", "Drug Plan Deductible", 
                     "Maximum Out-of-Pocket", "Health Premium", "Drug Premium", "Part B Premium"],
        plan1_details["Plan"]: [plan1_details["Star Rating"], plan1_details["Health Deductible"],
                                plan1_details["Drug Plan Deductible"], plan1_details["Maximum Out-of-Pocket"],
                                plan1_details["Health Premium"], plan1_details["Drug Premium"], 
                                plan1_details["Part B Premium"]],
        plan2_details["Plan"]: [plan2_details["Star Rating"], plan2_details["Health Deductible"],
                                plan2_details["Drug Plan Deductible"], plan2_details["Maximum Out-of-Pocket"],
                                plan2_details["Health Premium"], plan2_details["Drug Premium"],
                                plan2_details["Part B Premium"]]
    }
    
    # Create the main comparison DataFrame
    df_main = pd.DataFrame(comparison)
    
    # Features comparison
    features = []
    plan1_features = []
    plan2_features = []
    
    # All required feature names
    required_features = [
        "Vision", "Dental", "Hearing", "Transportation", "Fitness",
        "Worldwide Emergency", "Over The Counter", "In-Home Support", 
        "Emergency Response", "Bathroom Safety", "Meals", "Physical Exams",
        "Telehealth", "Endodontics", "Periodontics"
    ]
    
    # First add all required features
    for feature in required_features:
        features.append(feature)
        plan1_features.append(plan1_details["Features"].get(feature, "Not Found"))
        plan2_features.append(plan2_details["Features"].get(feature, "Not Found"))
    
    # Then add any additional features from either plan
    all_features = set(list(plan1_details["Features"].keys()) + list(plan2_details["Features"].keys()))
    for feature in sorted(all_features):
        if feature not in required_features:
            features.append(feature)
            plan1_features.append(plan1_details["Features"].get(feature, "Not Found"))
            plan2_features.append(plan2_details["Features"].get(feature, "Not Found"))
    
    df_features = pd.DataFrame({
        "Feature": features,
        plan1_details["Plan"]: plan1_features,
        plan2_details["Plan"]: plan2_features
    })
    
    # Benefits comparison
    benefits = []
    plan1_benefits = []
    plan2_benefits = []
    
    # Required benefit names
    required_benefits = [
        "Primary Doctor", "Specialist", "Diagnostic Tests", "Lab Services",
        "X-Rays", "Emergency Care", "Urgent Care",
        "Inpatient Hospital", "Outpatient Hospital", "Preventive Services"
    ]
    
    # First add all required benefits
    for benefit in required_benefits:
        benefits.append(benefit)
        plan1_benefits.append(plan1_details["Benefits"].get(benefit, "Not Found"))
        plan2_benefits.append(plan2_details["Benefits"].get(benefit, "Not Found"))
    
    # Then add any additional benefits from either plan
    all_benefits = set(list(plan1_details["Benefits"].keys()) + list(plan2_details["Benefits"].keys()))
    for benefit in sorted(all_benefits):
        if benefit not in required_benefits:
            benefits.append(benefit)
            plan1_benefits.append(plan1_details["Benefits"].get(benefit, "Not Found"))
            plan2_benefits.append(plan2_details["Benefits"].get(benefit, "Not Found"))
    
    df_benefits = pd.DataFrame({
        "Benefit": benefits,
        plan1_details["Plan"]: plan1_benefits,
        plan2_details["Plan"]: plan2_benefits
    })
    
    # Drug coverage comparison
    df_drugs = pd.DataFrame({
        "Coverage": ["Drug Coverage"],
        plan1_details["Plan"]: [plan1_details["Drug Coverage"]],
        plan2_details["Plan"]: [plan2_details["Drug Coverage"]]
    })
    
    return df_main, df_features, df_benefits, df_drugs

def extract_medical_conditions(question):
    """Extract medical conditions from user question"""
    # Dictionary of common medical conditions to look for
    conditions = [
        "diabetes", "heart disease", "asthma", "cancer", "arthritis", 
        "hypertension", "high blood pressure", "depression", "anxiety",
        "copd", "alzheimer", "dementia", "kidney disease", "stroke",
        "multiple sclerosis", "parkinson", "chronic pain"
    ]
    
    found_conditions = []
    normalized_question = question.lower()
    
    for condition in conditions:
        if condition in normalized_question:
            found_conditions.append(condition)
    
    # If no specific conditions found, extract potential health-related terms
    if not found_conditions:
        health_indicators = ["medicine", "prescription", "drug", "therapy", "specialist", 
                           "condition", "doctor", "hospital", "surgery", "medical"]
        
        for indicator in health_indicators:
            if indicator in normalized_question:
                found_conditions.append(indicator)
                break
    
    # Default to general health if no conditions found
    if not found_conditions:
        found_conditions = ["health care"]
    
    return found_conditions

def get_condition_related_services(conditions):
    """Get services related to specific medical conditions"""
    condition_services = {
        "diabetes": ["specialist", "endocrinologist", "lab", "drug", "supplies", "preventive"],
        "heart disease": ["cardiologist", "specialist", "diagnostic", "emergency", "hospital"],
        "asthma": ["specialist", "pulmonologist", "prescription", "emergency"],
        "cancer": ["oncologist", "specialist", "hospital", "radiation", "chemotherapy"],
        "arthritis": ["rheumatologist", "specialist", "therapy", "pain management"],
        "hypertension": ["specialist", "lab", "diagnostic", "preventive"],
        "depression": ["mental health", "therapy", "psychiatric", "prescription"],
        "anxiety": ["mental health", "therapy", "psychiatric", "prescription"],
        "chronic pain": ["pain management", "specialist", "therapy", "prescription"],
        "health care": ["primary care", "specialist", "preventive", "hospital", "emergency"]
    }
    
    services = []
    for condition in conditions:
        condition_key = next((k for k in condition_services.keys() if k in condition), "health care")
        services.extend(condition_services[condition_key])
    
    return list(set(services))  # Remove duplicates

def extract_cost_from_content(content):
    """Extract cost information from content text"""
    if not content:
        return "not found"
        
    # Look for dollar amounts
    matches = re.findall(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', content.lower())
    if matches:
        return matches[0]
    
    # Check for zero cost indicators
    if any(term in content.lower() for term in ["$0", "no copay", "0 copay", "fully covered"]):
        return "0"
    
    # Check if covered or not covered
    if "covered" in content.lower() and "not covered" not in content.lower():
        return "covered"
    elif "not covered" in content.lower():
        return "not covered"
    
    return "not found"

def compare_and_recommend(results, plan1_title, plan2_title, conditions):
    """Compare results and generate a recommendation based on plan coverage for conditions"""
    plan1_data = results[plan1_title]
    plan2_data = results[plan2_title]
    
    comparison_points = []
    plan1_score = 0
    plan2_score = 0
    
    # Compare direct condition mentions
    for condition in conditions:
        if condition in plan1_data and condition in plan2_data:
            # Both plans mention the condition
            plan1_mentions = len(plan1_data[condition])
            plan2_mentions = len(plan2_data[condition])
            
            if plan1_mentions > plan2_mentions:
                comparison_points.append(f"{plan1_title} has more detailed information about {condition} care.")
                plan1_score += 1
            elif plan2_mentions > plan1_mentions:
                comparison_points.append(f"{plan2_title} has more detailed information about {condition} care.")
                plan2_score += 1
        elif condition in plan1_data:
            comparison_points.append(f"Only {plan1_title} specifically mentions {condition} care.")
            plan1_score += 2
        elif condition in plan2_data:
            comparison_points.append(f"Only {plan2_title} specifically mentions {condition} care.")
            plan2_score += 2
    
    # Compare service costs
    service_keys = [k for k in plan1_data.keys() if k.endswith('_services')] + [k for k in plan2_data.keys() if k.endswith('_services')]
    service_keys = list(set(service_keys))  # Remove duplicates
    
    for service_key in service_keys:
        service_name = service_key.replace('_services', '')
        
        plan1_service = plan1_data.get(service_key, [])
        plan2_service = plan2_data.get(service_key, [])
        
        if plan1_service and plan2_service:
            # Compare costs if available
            plan1_cost = extract_cost(plan1_service[0].get('cost', 'not found'))
            plan2_cost = extract_cost(plan2_service[0].get('cost', 'not found'))
            
            if plan1_cost is not None and plan2_cost is not None:
                if plan1_cost < plan2_cost:
                    comparison_points.append(f"{plan1_title} has lower costs for {service_name} services (${plan1_cost} vs ${plan2_cost}).")
                    plan1_score += 2
                elif plan2_cost < plan1_cost:
                    comparison_points.append(f"{plan2_title} has lower costs for {service_name} services (${plan2_cost} vs ${plan1_cost}).")
                    plan2_score += 2
            elif plan1_cost is not None:
                comparison_points.append(f"{plan1_title} has defined costs for {service_name} services (${plan1_cost}).")
                plan1_score += 1
            elif plan2_cost is not None:
                comparison_points.append(f"{plan2_title} has defined costs for {service_name} services (${plan2_cost}).")
                plan2_score += 1
        elif plan1_service:
            comparison_points.append(f"{plan1_title} specifically covers {service_name} services.")
            plan1_score += 1
        elif plan2_service:
            comparison_points.append(f"{plan2_title} specifically covers {service_name} services.")
            plan2_score += 1
    
    # Generate recommendation
    recommendation = "Based on the available information:\n\n"
    
    for point in comparison_points[:5]:  # Limit to top 5 points
        recommendation += f"- {point}\n"
    
    if plan1_score > plan2_score:
        recommendation += f"\nOverall, {plan1_title} appears to be better suited for your needs with {conditions[0]}."
    elif plan2_score > plan1_score:
        recommendation += f"\nOverall, {plan2_title} appears to be better suited for your needs with {conditions[0]}."
    else:
        recommendation += f"\nBoth plans appear similar in their coverage for {conditions[0]}. Consider other factors like provider networks."
    
    return recommendation

def extract_cost(cost_str):
    """Extract numeric cost from cost string"""
    if cost_str == "0" or cost_str == "covered":
        return 0
    
    try:
        if cost_str.isdigit():
            return int(cost_str)
        
        if cost_str.startswith("$"):
            cost_value = cost_str.replace("$", "").replace(",", "")
            return float(cost_value)
    except:
        pass
    
    return None

def ask_plan_question(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title, question):
    """Process a user question about plans and generate a recommendation"""
    # Extract conditions from the question
    conditions = extract_medical_conditions(question)
    print(f"Analyzing for conditions: {', '.join(conditions)}")
    
    # Get related services for these conditions
    services = get_condition_related_services(conditions)
    print(f"Related services: {', '.join(services[:5])}...")
    
    # Directly get benefit information for each plan
    plan1_benefits = extract_benefit_info(neo4j_uri, neo4j_user, neo4j_password, plan1_title)
    plan2_benefits = extract_benefit_info(neo4j_uri, neo4j_user, neo4j_password, plan2_title)
    
    # Get feature information for each plan
    plan1_features = detect_features_from_content(neo4j_uri, neo4j_user, neo4j_password, plan1_title)
    plan2_features = detect_features_from_content(neo4j_uri, neo4j_user, neo4j_password, plan2_title)
    
    # Create search results dictionary for both plans
    results = {
        plan1_title: {"benefits": plan1_benefits, "features": plan1_features},
        plan2_title: {"benefits": plan2_benefits, "features": plan2_features}
    }
    
    # For each condition, search in both plans
    for condition in conditions:
        # Search plan 1
        plan1_results = search_section_content(neo4j_uri, neo4j_user, neo4j_password, 
                                             plan1_title, condition)
        if plan1_results:
            results[plan1_title][condition] = plan1_results
            
        # Search plan 2
        plan2_results = search_section_content(neo4j_uri, neo4j_user, neo4j_password, 
                                             plan2_title, condition)
        if plan2_results:
            results[plan2_title][condition] = plan2_results
    
    # For each service, search in both plans
    for service in services:
        service_key = f"{service.replace(' ', '_')}_services"
        
        # Search plan 1
        plan1_results = search_section_content(neo4j_uri, neo4j_user, neo4j_password, 
                                             plan1_title, service)
        if plan1_results:
            results[plan1_title][service_key] = [{"content": r["content"], 
                                                 "cost": extract_cost_from_content(r["content"])} 
                                                for r in plan1_results]
            
        # Search plan 2
        plan2_results = search_section_content(neo4j_uri, neo4j_user, neo4j_password, 
                                             plan2_title, service)
        if plan2_results:
            results[plan2_title][service_key] = [{"content": r["content"], 
                                                 "cost": extract_cost_from_content(r["content"])} 
                                                for r in plan2_results]
    
    # Generate recommendation based on comparison
    recommendation = balanced_comparison(results, plan1_title, plan2_title, conditions, services)
    return recommendation

def balanced_comparison(results, plan1_title, plan2_title, conditions, services):
    """Compare results and generate a balanced recommendation"""
    plan1_data = results[plan1_title]
    plan2_data = results[plan2_title]
    
    comparison_points = []
    plan1_score = 0
    plan2_score = 0
    
    # Track specific comparison metrics for explanation
    plan1_matches = 0
    plan2_matches = 0
    plan1_cost_wins = 0
    plan2_cost_wins = 0
    
    # Compare direct condition mentions
    for condition in conditions:
        if condition in plan1_data and condition in plan2_data:
            # Both plans mention the condition
            plan1_mentions = len(plan1_data[condition])
            plan2_mentions = len(plan2_data[condition])
            
            if plan1_mentions > plan2_mentions:
                comparison_points.append(f"{plan1_title} mentions {condition} more frequently ({plan1_mentions} vs {plan2_mentions} times).")
                plan1_score += 1
                plan1_matches += 1
            elif plan2_mentions > plan1_mentions:
                comparison_points.append(f"{plan2_title} mentions {condition} more frequently ({plan2_mentions} vs {plan1_mentions} times).")
                plan2_score += 1
                plan2_matches += 1
            else:
                comparison_points.append(f"Both plans mention {condition} equally ({plan1_mentions} times).")
        elif condition in plan1_data:
            comparison_points.append(f"Only {plan1_title} specifically mentions {condition}.")
            plan1_score += 2
            plan1_matches += 1
        elif condition in plan2_data:
            comparison_points.append(f"Only {plan2_title} specifically mentions {condition}.")
            plan2_score += 2
            plan2_matches += 1
    
    # Compare related services
    relevant_benefits = []
    for service in services:
        # Find matching benefits
        service_words = service.split()
        for benefit in plan1_data.get("benefits", {}):
            if any(word.lower() in benefit.lower() for word in service_words):
                relevant_benefits.append(benefit)
    
    relevant_benefits = list(set(relevant_benefits))  # Remove duplicates
    
    # Compare costs for relevant benefits
    for benefit in relevant_benefits:
        plan1_cost = extract_numeric_cost(plan1_data.get("benefits", {}).get(benefit, "Not Found"))
        plan2_cost = extract_numeric_cost(plan2_data.get("benefits", {}).get(benefit, "Not Found"))
        
        if plan1_cost is not None and plan2_cost is not None:
            if plan1_cost < plan2_cost:
                comparison_points.append(f"{plan1_title} has lower {benefit} costs (${plan1_cost} vs ${plan2_cost}).")
                plan1_score += 2
                plan1_cost_wins += 1
            elif plan2_cost < plan1_cost:
                comparison_points.append(f"{plan2_title} has lower {benefit} costs (${plan2_cost} vs ${plan1_cost}).")
                plan2_score += 2
                plan2_cost_wins += 1
            else:
                comparison_points.append(f"Both plans have equal {benefit} costs (${plan1_cost}).")
        elif plan1_cost is not None:
            comparison_points.append(f"{plan1_title} specifies {benefit} costs (${plan1_cost}), but {plan2_title} doesn't.")
            plan1_score += 1
        elif plan2_cost is not None:
            comparison_points.append(f"{plan2_title} specifies {benefit} costs (${plan2_cost}), but {plan1_title} doesn't.")
            plan2_score += 1
    
    # Compare service costs from direct searches
    service_keys = [k for k in plan1_data.keys() if k.endswith('_services')] + [k for k in plan2_data.keys() if k.endswith('_services')]
    service_keys = list(set(service_keys))  # Remove duplicates
    
    for service_key in service_keys:
        service_name = service_key.replace('_services', '').replace('_', ' ')
        
        plan1_service = plan1_data.get(service_key, [])
        plan2_service = plan2_data.get(service_key, [])
        
        if plan1_service and plan2_service:
            # Compare costs if available
            plan1_cost = extract_numeric_cost(plan1_service[0].get('cost', 'not found'))
            plan2_cost = extract_numeric_cost(plan2_service[0].get('cost', 'not found'))
            
            if plan1_cost is not None and plan2_cost is not None:
                if plan1_cost < plan2_cost:
                    comparison_points.append(f"{plan1_title} has lower {service_name} costs (${plan1_cost} vs ${plan2_cost}).")
                    plan1_score += 2
                    plan1_cost_wins += 1
                elif plan2_cost < plan1_cost:
                    comparison_points.append(f"{plan2_title} has lower {service_name} costs (${plan2_cost} vs ${plan1_cost}).")
                    plan2_score += 2
                    plan2_cost_wins += 1
                else:
                    comparison_points.append(f"Both plans have equal {service_name} costs (${plan1_cost}).")
            elif plan1_cost is not None:
                comparison_points.append(f"{plan1_title} specifies {service_name} costs (${plan1_cost}), but {plan2_title} doesn't.")
                plan1_score += 1
            elif plan2_cost is not None:
                comparison_points.append(f"{plan2_title} specifies {service_name} costs (${plan2_cost}), but {plan1_title} doesn't.")
                plan2_score += 1
        elif plan1_service:
            comparison_points.append(f"Only {plan1_title} mentions {service_name} services.")
            plan1_score += 1
            plan1_matches += 1
        elif plan2_service:
            comparison_points.append(f"Only {plan2_title} mentions {service_name} services.")
            plan2_score += 1
            plan2_matches += 1
    
    # Ensure we have a good set of comparison points
    if len(comparison_points) < 3:
        # Add general comparison points if specific ones are limited
        plan1_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan1_title)
        plan2_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan2_title)
        
        # Compare general costs
        for cost_type in ["Health Deductible", "Drug Plan Deductible", "Maximum Out-of-Pocket"]:
            plan1_cost = extract_numeric_cost(plan1_details.get(cost_type, "Not found"))
            plan2_cost = extract_numeric_cost(plan2_details.get(cost_type, "Not found"))
            
            if plan1_cost is not None and plan2_cost is not None:
                if plan1_cost < plan2_cost:
                    comparison_points.append(f"{plan1_title} has lower {cost_type} (${plan1_cost} vs ${plan2_cost}).")
                    plan1_score += 1
                elif plan2_cost < plan1_cost:
                    comparison_points.append(f"{plan2_title} has lower {cost_type} (${plan2_cost} vs ${plan1_cost}).")
                    plan2_score += 1
    
    # Generate recommendation
    recommendation = f"Based on the analysis for {', '.join(conditions)}:\n\n"
    
    # Take the top comparison points (up to 5)
    used_points = []
    for point in comparison_points:
        if len(used_points) >= 5:
            break
        
        # Avoid duplicate types of points
        if not any(p.split(" has lower ")[0] == point.split(" has lower ")[0] for p in used_points if " has lower " in p):
            used_points.append(point)
    
    for point in used_points:
        recommendation += f"- {point}\n"
    
    # Add score summary
    if len(used_points) < len(comparison_points):
        recommendation += f"\nI analyzed {len(comparison_points)} total comparison points.\n"
    
    recommendation += f"\nAnalysis summary:"
    recommendation += f"\n- {plan1_title}: {plan1_matches} relevant matches, {plan1_cost_wins} cost advantages"
    recommendation += f"\n- {plan2_title}: {plan2_matches} relevant matches, {plan2_cost_wins} cost advantages"
    
    if plan1_score > plan2_score:
        recommendation += f"\n\nBased on this analysis, {plan1_title} appears to be better suited for {conditions[0]}."
    elif plan2_score > plan1_score:
        recommendation += f"\n\nBased on this analysis, {plan2_title} appears to be better suited for {conditions[0]}."
    else:
        recommendation += f"\n\nBoth plans appear similar in their coverage for {conditions[0]}. Consider other factors like provider networks."
    
    return recommendation

def extract_numeric_cost(cost_str):
    """Extract numeric cost from cost string with improved handling"""
    if not cost_str or cost_str == "Not Found" or cost_str == "not found":
        return None
    
    if cost_str == "$0" or cost_str == "0" or cost_str == "Covered" or cost_str == "covered":
        return 0
    
    try:
        if isinstance(cost_str, (int, float)):
            return float(cost_str)
            
        # Handle string formats
        if cost_str.isdigit():
            return float(cost_str)
        
        if cost_str.startswith("$"):
            cost_value = cost_str.replace("$", "").replace(",", "")
            return float(cost_value)
    except:
        pass
    
    return None

def interactive_qa_mode(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title):
    """Interactive QA mode for comparing plans for specific needs"""
    print("\nPlan Comparison Q&A Mode")
    print("Ask questions like 'Which plan is better for diabetes?' or 'If I need regular specialist visits, which plan should I choose?'")
    print("Type 'exit' to return to main menu\n")
    
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
        
        print("Analyzing plans...")
        recommendation = ask_plan_question(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title, question)
        
        print("\nRecommendation:")
        print(recommendation)
        print("\n---")

def compare_medicare_plans(neo4j_uri, neo4j_user, neo4j_password):
    """Main function for comparing Medicare plans"""
    print("\nMedicare Plan Comparison Tool")
    
    # List available documents
    available_docs = list_documents(neo4j_uri, neo4j_user, neo4j_password)
    if len(available_docs) < 2:
        print("Need at least 2 documents for comparison. Please parse more Medicare plan documents.")
        return
    
    # Display documents
    print("\nAvailable plans:")
    for i, doc in enumerate(available_docs):
        print(f"{i+1}. {doc}")
    
    # Select two plans
    try:
        plan1_idx = int(input("\nSelect first plan (number): ")) - 1
        plan2_idx = int(input("Select second plan (number): ")) - 1
        
        if not (0 <= plan1_idx < len(available_docs) and 0 <= plan2_idx < len(available_docs)):
            print("Invalid selection. Please try again.")
            return
        
        plan1_title = available_docs[plan1_idx]
        plan2_title = available_docs[plan2_idx]
        
        if plan1_title == plan2_title:
            print("Please select two different plans.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    print(f"\nComparing: {plan1_title} vs {plan2_title}")
    print("Extracting plan details (this may take a moment)...")
    
    # Extract details for both plans
    plan1_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan1_title)
    plan2_details = extract_plan_details(neo4j_uri, neo4j_user, neo4j_password, plan2_title)
    
    # Format the comparison
    df_main, df_features, df_benefits, df_drugs = format_comparison(plan1_details, plan2_details)
    
    # Display the comparison
    print("\n== PLAN COMPARISON ==\n")
    print("\nMain Plan Details:")
    print(tabulate(df_main, headers='keys', tablefmt='grid', showindex=False))
    
    print("\nFeatures:")
    print(tabulate(df_features, headers='keys', tablefmt='grid', showindex=False))
    
    print("\nBenefits and Costs:")
    print(tabulate(df_benefits, headers='keys', tablefmt='grid', showindex=False))
    
    print("\nDrug Coverage:")
    print(tabulate(df_drugs, headers='keys', tablefmt='grid', showindex=False))
    
    print("\nNote: 'Not found' indicates the information couldn't be extracted from the document.")
    
    # Ask if user would like to enter QA mode
    qa_choice = input("\nWould you like to ask specific questions about these plans? (y/n): ")
    if qa_choice.lower() == 'y':
        interactive_qa_mode(neo4j_uri, neo4j_user, neo4j_password, plan1_title, plan2_title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Medicare Plans')
    parser.add_argument('--uri', required=True, help='Neo4j connection URI')
    parser.add_argument('--user', required=True, help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    
    args = parser.parse_args()
    
    compare_medicare_plans(args.uri, args.user, args.password)