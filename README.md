### bedrock_rag_sample
How it Works & Key Bedrock Parts:
bedrock_runtime.invoke_model(...): This is the core Boto3 call to interact with any Bedrock model.
modelId: Specifies which Bedrock model to use (e.g., amazon.titan-embed-text-v1).
body: The input payload for the model, formatted as JSON. The structure of this body varies depending on the model.
For Titan Embeddings (amazon.titan-embed-text-v1):
{
    "inputText": "Your text to embed"
}
Use code with caution.
Json
The response contains an "embedding" key with a list of floats.
For LLMs like Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0) - Messages API:
{
    "anthropic_version": "bedrock-2023-05-31", // Required for Claude 3
    "max_tokens": 1000, // Or other desired value
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Your detailed prompt here..."}]
        }
        // Optionally add "assistant" role messages for few-shot prompting
    ]
}
Use code with caution.
Json
The response (for Claude Messages API) contains a "content" list, and the generated text is typically in content[0].text.
accept: Typically "application/json".
contentType: Typically "application/json".
Embedding (get_embedding_titan):
Takes text as input.
Calls invoke_model with the Titan embedding model ID.
Parses the JSON response to extract the numerical embedding vector.
Entity Recognition (get_entities_from_llm):
This function uses a general-purpose LLM (like Claude 3 Sonnet) to perform NER.
Prompt Engineering is Key: The prompt instructs the LLM:
What task to perform (extract named entities).
What types of entities to look for.
Crucially, it asks for the output in a structured JSON format. This makes parsing the LLM's response much easier and more reliable than trying to parse free-form text.
An example is provided in the prompt (few-shot learning) to guide the model.
The LLM's text output (which should be the JSON string) is then parsed using json.loads().
Note: Using an LLM for NER is flexible but might be less accurate or consistent than dedicated NER services like Amazon Comprehend, especially for complex or domain-specific entities. However, it demonstrates the capability within Bedrock.
Retrieval & Augmentation: These parts are conceptually similar to a non-Bedrock RAG, but they use the embeddings and entities derived from Bedrock services.
To Run This Code:
Save it as a Python file (e.g., bedrock_rag_sample.py).
Ensure your AWS environment is set up (credentials, region).
Verify model access in the Bedrock console for your region.
Update AWS_REGION if necessary.
Run python bedrock_rag_sample.py.
