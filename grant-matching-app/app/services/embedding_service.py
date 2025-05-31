from openai import AzureOpenAI
import config

client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    api_base=config.AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15"
)

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding
