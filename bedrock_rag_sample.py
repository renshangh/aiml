import boto3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
AWS_REGION = "us-east-1"  # IMPORTANT: Change to your Bedrock region
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
# For NER, we'll use a powerful instruction-following model.
# Claude 3 Sonnet is a good choice. Other models like AI21 J2 or Cohere Command could also work.
NER_LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" # Use Claude 3 Haiku for faster/cheaper option if available & sufficient
# NER_LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # Potentially faster/cheaper
# NER_LLM_MODEL_ID = "ai21.j2-grande-instruct" # Alternative

# Initialize the Bedrock runtime client
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )
    print(f"Successfully connected to Bedrock runtime in {AWS_REGION}")
except Exception as e:
    print(f"Error connecting to Bedrock runtime: {e}")
    print("Please ensure your AWS credentials and region are configured correctly,")
    print(f"and you have model access for {EMBEDDING_MODEL_ID} and {NER_LLM_MODEL_ID} in {AWS_REGION}.")
    exit()

# --- 1. Sample Knowledge Base ---
print("\n--- 1. Sample Knowledge Base ---")
knowledge_base_texts = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, designed by Gustave Eiffel.",
    "Paris is the capital and most populous city of France, known for its art, fashion, and culture. The mayor is Anne Hidalgo.",
    "The Louvre Museum, located in Paris, is the world's largest art museum and a historic monument.",
    "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
    "Tim Cook is the CEO of Apple, succeeding Steve Jobs. Apple was founded on April 1, 1976."
]
print(f"Knowledge base has {len(knowledge_base_texts)} documents.")

# --- 2. Embedding with Amazon Titan ---
print("\n--- 2. Embedding with Amazon Titan ---")

def get_embedding_titan(text_input, model_id=EMBEDDING_MODEL_ID):
    """
    Generates an embedding for the given text using Amazon Titan Text Embeddings.
    """
    body = json.dumps({
        "inputText": text_input
    })
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        return embedding
    except Exception as e:
        print(f"Error getting embedding for '{text_input[:30]}...': {e}")
        return None

# Embed the knowledge base
knowledge_base_embeddings = []
for i, text in enumerate(knowledge_base_texts):
    print(f"Embedding document {i+1}/{len(knowledge_base_texts)}...")
    embedding = get_embedding_titan(text)
    if embedding:
        knowledge_base_embeddings.append(embedding)
    else:
        # Handle cases where embedding might fail (e.g. empty string, model issue)
        knowledge_base_embeddings.append(None) # Or skip, or add a zero vector

# Filter out None embeddings if any failed
knowledge_base_embeddings = [emb for emb in knowledge_base_embeddings if emb is not None]
if knowledge_base_embeddings:
    knowledge_base_embeddings_np = np.array(knowledge_base_embeddings)
    print(f"Shape of knowledge base embeddings: {knowledge_base_embeddings_np.shape}")
    print(f"Sample embedding for the first document (first 5 dimensions): {knowledge_base_embeddings_np[0][:5]}")
else:
    print("No embeddings were generated for the knowledge base.")
    knowledge_base_embeddings_np = np.array([])


# --- 3. User Query & Embedding ---
print("\n--- 3. User Query & Embedding ---")
user_query = "Who is the CEO of Apple and where are its headquarters?"
print(f"User query: '{user_query}'")

query_embedding = get_embedding_titan(user_query)
if query_embedding:
    query_embedding_np = np.array([query_embedding]) # Needs to be 2D for cosine_similarity
    print(f"Shape of query embedding: {query_embedding_np.shape}")
    print(f"Sample query embedding (first 5 dimensions): {query_embedding_np[0][:5]}")
else:
    print("Failed to generate embedding for the query.")
    query_embedding_np = np.array([])


# --- 4. Retrieval (Simplified Semantic Search) ---
# This part is similar to the previous example, using the Bedrock embeddings.
print("\n--- 4. Retrieval using Embeddings ---")
retrieved_contexts = []
if query_embedding_np.size > 0 and knowledge_base_embeddings_np.size > 0:
    similarities = cosine_similarity(query_embedding_np, knowledge_base_embeddings_np)
    top_k = 2
    # Ensure we don't ask for more results than available documents
    actual_top_k = min(top_k, len(knowledge_base_texts))
    if actual_top_k > 0:
        top_k_indices = np.argsort(similarities[0])[-actual_top_k:][::-1]

        print(f"Top {actual_top_k} relevant documents for query: '{user_query}'")
        for i, idx in enumerate(top_k_indices):
            context = knowledge_base_texts[idx] # Assuming order is preserved
            similarity_score = similarities[0][idx]
            retrieved_contexts.append(context)
            print(f"  {i+1}. (Score: {similarity_score:.4f}) {context}")
    else:
        print("No documents in knowledge base to retrieve from.")
else:
    print("Cannot perform retrieval due to missing query or knowledge base embeddings.")


# --- 5. Entity Recognition using a Bedrock LLM (e.g., Claude 3 Sonnet) ---
print("\n--- 5. Entity Recognition (NER) with Bedrock LLM ---")

def get_entities_from_llm(text_input, model_id=NER_LLM_MODEL_ID):
    """
    Uses a Bedrock LLM to extract entities from text.
    This is a prompted approach and may not be as robust as dedicated NER services
    like Amazon Comprehend for all use cases but demonstrates LLM capability.
    """
    prompt = f"""\
Please extract named entities from the following text.
Identify entities like PERSON, ORGANIZATION, LOCATION, DATE, and PRODUCT.
For each entity, provide its text and its type.
Return the output as a JSON list of objects, where each object has a "text" and "type" key.
Example:
Text: "Apple announced the new iPhone in Cupertino on September 12th."
Output:
[
  {{"text": "Apple", "type": "ORGANIZATION"}},
  {{"text": "iPhone", "type": "PRODUCT"}},
  {{"text": "Cupertino", "type": "LOCATION"}},
  {{"text": "September 12th", "type": "DATE"}}
]

Text to analyze: "{text_input}"
Output:
"""

    # Construct the body according to the model's expected input format
    if "anthropic.claude" in model_id:
        # Claude Messages API
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }
    elif "ai21" in model_id: # Example for AI21
        payload = {
            "prompt": prompt,
            "maxTokens": 500,
            "temperature": 0.1,
            "stopSequences": [] # Adjust as needed
        }
    # Add other model families if needed (e.g., cohere)
    else:
        print(f"Model family for {model_id} not explicitly handled for payload construction. Using generic text.")
        # This might not work correctly for all models.
        payload = {"inputText": prompt, "textGenerationConfig": {"maxTokenCount": 512}}


    body = json.dumps(payload)

    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())

        # Parse the response based on the model
        if "anthropic.claude" in model_id:
            llm_output_text = response_body.get("content")[0].get("text")
        elif "ai21" in model_id:
            llm_output_text = response_body.get("completions")[0].get("data").get("text")
        else: # Fallback, assuming a simple text output
            llm_output_text = response_body.get("results")[0].get("outputText") # Adjust if needed

        # The LLM should return a JSON string based on the prompt.
        # We need to extract this JSON string carefully.
        # Sometimes LLMs add introductory text like "Here is the JSON output:"
        json_start_index = llm_output_text.find('[')
        json_end_index = llm_output_text.rfind(']') + 1
        if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
            json_str = llm_output_text[json_start_index:json_end_index]
            entities = json.loads(json_str)
            return entities
        else:
            print(f"Could not extract valid JSON from LLM output for NER: {llm_output_text}")
            return []

    except Exception as e:
        print(f"Error getting entities for '{text_input[:30]}...': {e}")
        return []

# a) NER on the user query
print("\na) Entities in User Query:")
query_entities = get_entities_from_llm(user_query)
if query_entities:
    for entity in query_entities:
        print(f"  - Entity: '{entity.get('text')}', Type: '{entity.get('type')}'")
else:
    print("  No entities found or error in processing the query.")

# b) NER on the most relevant retrieved context
print("\nb) Entities in the Most Relevant Retrieved Context (if any):")
if retrieved_contexts:
    most_relevant_context = retrieved_contexts[0]
    print(f"Context: \"{most_relevant_context}\"")
    context_entities = get_entities_from_llm(most_relevant_context)
    if context_entities:
        for entity in context_entities:
            print(f"  - Entity: '{entity.get('text')}', Type: '{entity.get('type')}'")
    else:
        print("  No entities found or error in processing the most relevant context.")
else:
    print("  No context was retrieved, skipping NER on context.")


# --- 6. Augmentation (Conceptual for RAG) ---
print("\n--- 6. Augmentation (Conceptual) ---")
# In a full RAG, you'd combine the query and retrieved_contexts for an LLM.
# The extracted entities could also be used to refine the prompt or highlight key info.

if retrieved_contexts:
    augmented_prompt_parts = [f"User Query: {user_query}\n"]
    if query_entities:
        augmented_prompt_parts.append("Entities in Query:")
        for entity in query_entities:
            augmented_prompt_parts.append(f"  - {entity.get('text')} ({entity.get('type')})")
        augmented_prompt_parts.append("\n")


    augmented_prompt_parts.append("Relevant Context from Knowledge Base:")
    for i, context in enumerate(retrieved_contexts):
        augmented_prompt_parts.append(f"--- Context {i+1} ---")
        augmented_prompt_parts.append(context)
        # Optionally add entities from this context too
        # context_entities_for_prompt = get_entities_from_llm(context)
        # if context_entities_for_prompt:
        #     augmented_prompt_parts.append("Entities in this context:")
        #     for entity in context_entities_for_prompt:
        #         augmented_prompt_parts.append(f"  - {entity.get('text')} ({entity.get('type')})")
        augmented_prompt_parts.append("---")

    augmented_prompt_parts.append("\nBased on the provided context and query, please answer the user query.")
    augmented_prompt_parts.append("Answer:")
    final_augmented_prompt = "\n".join(augmented_prompt_parts)

    print("Augmented Prompt (simplified example for LLM):\n")
    print(final_augmented_prompt)
else:
    print("No context retrieved, cannot create augmented prompt for LLM.")

print("\n--- End of Demo ---")