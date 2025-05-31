resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
}

# Azure Storage Account
resource "azurerm_storage_account" "main" {
  name                     = "${var.app_name}sa"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "grantdocs" {
  name                  = "grant-docs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Azure Cognitive Search
resource "azurerm_search_service" "main" {
  name                = "${var.app_name}-search"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "basic"
  partition_count     = 1
  replica_count       = 1
}

# App Service Plan
resource "azurerm_app_service_plan" "main" {
  name                = "${var.app_name}-plan"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "Linux"
  reserved            = true

  sku {
    tier = "Basic"
    size = "B1"
  }
}

# Web App
resource "azurerm_linux_web_app" "main" {
  name                = var.app_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  service_plan_id     = azurerm_app_service_plan.main.id

  site_config {
    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = {
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE" = "false"
    "AZURE_SEARCH_ENDPOINT"               = azurerm_search_service.main.query_keys[0].key
    "AZURE_STORAGE_ACCOUNT"               = azurerm_storage_account.main.name
  }
}

