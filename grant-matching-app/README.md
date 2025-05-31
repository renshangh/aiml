
# 🧠 Generative AI Grant Matching System (Azure + Flask + Terraform)

This project enables automated matching between research proposals and funding grants using Azure OpenAI and Azure Cognitive Search, with a Flask-based web interface. It’s fully deployable with Terraform and runs on Azure App Service.

---

## 📦 Features

- 🔍 **Semantic Grant Matching** using Azure OpenAI embeddings + Azure Cognitive Search
- 💬 **Generative Summaries** to explain why a grant fits a proposal
- ☁️ **Azure-native deployment** (App Service, Blob Storage, Cognitive Search)
- 🛠️ Infrastructure-as-Code using **Terraform**

---

## 🚀 Deployment Overview

### ☁️ 1. Prerequisites

- Azure CLI installed and authenticated
- Terraform v1.3+ installed
- Access to an Azure subscription with:
  - Cognitive Search and Web App quotas
  - [Azure OpenAI access](https://oai.azure.com/portal) (manual approval needed)

---

### 📁 2. Setup Instructions

#### Step A: Clone & Initialize

```bash
git clone https://github.com/your-org/grant-matching-app.git
cd grant-matching-app/terraform
terraform init
```

#### Step B: Deploy Azure Infrastructure

```bash
terraform apply
```

This will provision:

- Azure Resource Group
- Azure Blob Storage (for grant documents)
- Azure Cognitive Search
- Azure App Service Plan & Linux Web App

> 📌 **Note**: The output will include the App Service URL and resource names.

---

### ⚙️ 3. Azure OpenAI (Manual Setup)

1. Navigate to [Azure OpenAI Studio](https://oai.azure.com/portal).
2. Create a resource with:
   - `GPT-35-Turbo` (for summaries)
   - `text-embedding-ada-002` (for semantic search)
3. Get:
   - Endpoint URL
   - API Key

4. Add these as **Application Settings** for the Web App via Azure Portal or CLI:

```bash
AZURE_OPENAI_API_KEY       = <your-openai-api-key>
AZURE_OPENAI_ENDPOINT      = https://<your-openai>.openai.azure.com/
AZURE_SEARCH_ENDPOINT      = https://<your-search>.search.windows.net/
AZURE_SEARCH_KEY           = <admin-or-query-key>
AZURE_SEARCH_INDEX_NAME    = grant-index
AZURE_BLOB_CONTAINER       = grant-docs
```

---

### 🐍 4. Deploy Flask App

You can deploy your Flask app (in `/grant-matcher`) in any of the following ways:

#### Option A: Azure CLI

```bash
az webapp up --name grant-matching-app   --resource-group grant-matching-rg   --runtime "PYTHON:3.11"
```

#### Option B: Deploy via GitHub Actions

Connect your repo via the Azure Portal for CI/CD-based deployment.

---

## 📄 App Usage

1. Navigate to the deployed App Service URL.
2. Paste a **research proposal** into the text input form.
3. Click Submit.

Behind the scenes:
- The proposal is embedded using OpenAI
- Similar grant documents are fetched via Azure Cognitive Search
- A GPT-generated summary explains why the grant is a good match

---

## 🧪 Local Testing

To run locally:

```bash
cd grant-matcher
export FLASK_APP=run.py
export FLASK_ENV=development
python run.py
```

Ensure the following environment variables are set locally:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_KEY`
- `AZURE_SEARCH_INDEX_NAME`

---

## 🗃 Optional Enhancements

- ✅ Add PDF parsing + OCR for uploaded documents
- ✳️ Use LangChain to chain reasoning + querying
- 🔁 Feedback loop stored in CosmosDB or Azure Table for iterative learning
- 📊 Log search usage for evaluation or future RLHF

---

## 📬 Support

Open a GitHub Issue or contact `team@synapsedynamics.ai` with questions or feedback.

---
