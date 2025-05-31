from openai import AzureOpenAI
import config

client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    api_base=config.AZURE_OPENAI_ENDPOINT,
    api_version="2023-05-15"
)

def generate_summary(proposal_text, grant_info):
    prompt = f"""Given this research proposal:\n{proposal_text}\n
    and this matched grant:\n{grant_info}\n
    Write a summary and suggest why this grant fits."""
    
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
