"""Tests for config module."""
import os
import pytest
from issues_agent.config import load_config


def test_config_loads_from_env(monkeypatch):
    """Test that config loads successfully from environment variables."""
    monkeypatch.setenv("GITHUB_TOKEN", "test_github_token")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test_deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    config = load_config()
    
    assert config.github_token == "test_github_token"
    assert config.azure_openai_endpoint == "https://test.openai.azure.com"
    assert config.azure_openai_api_key == "test_api_key"
    assert config.azure_openai_deployment == "test_deployment"
    assert config.azure_openai_api_version == "2024-02-15-preview"


def test_config_missing_github_token_raises_error(monkeypatch):
    """Test that missing GITHUB_TOKEN raises ValueError."""
    # Mock os.path.exists to prevent .env loading
    import issues_agent.config
    monkeypatch.setattr(issues_agent.config.os.path, "exists", lambda x: False)
    
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test_deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    with pytest.raises(ValueError) as exc_info:
        load_config()
    
    assert "GITHUB_TOKEN" in str(exc_info.value)


def test_config_missing_azure_vars_raises_error(monkeypatch):
    """Test that missing Azure OpenAI variables raise ValueError."""
    # Mock os.path.exists to prevent .env loading
    import issues_agent.config
    monkeypatch.setattr(issues_agent.config.os.path, "exists", lambda x: False)
    
    monkeypatch.setenv("GITHUB_TOKEN", "test_github_token")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test_deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    with pytest.raises(ValueError) as exc_info:
        load_config()
    
    assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)
