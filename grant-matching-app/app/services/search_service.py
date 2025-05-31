from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import config

search_client = SearchClient(
    endpoint=config.AZURE_SEARCH_ENDPOINT,
    index_name=config.AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
)

def search_similar_grants(embedding):
    vector_query = {
        "vector": embedding,
        "topK": 5,
        "fields": "contentVector",
        "filter": None
    }
    return search_client.search(search_text=None, vector_queries=[vector_query])
