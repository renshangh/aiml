output "web_app_url" {
  value = azurerm_linux_web_app.main.default_hostname
}

output "storage_account_name" {
  value = azurerm_storage_account.main.name
}

output "cognitive_search_name" {
  value = azurerm_search_service.main.name
}
