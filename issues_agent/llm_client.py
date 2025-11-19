"""Shared Azure OpenAI client for the issues agent."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Shared Azure OpenAI client wrapper.
    
    Provides a lazily-initialized client that can be reused across
    different components (classifier, duplicate detector, etc.).
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str,
        client: Optional[object] = None,
    ):
        """Initialize the Azure OpenAI client wrapper.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: API key for authentication
            deployment: Deployment/model name
            api_version: API version string
            client: Optional pre-initialized client (for testing)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self._client = client
    
    def _get_client(self):
        """Instantiate (or return injected) Azure OpenAI client lazily."""
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI  # lazy import
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("AzureOpenAI SDK not installed") from e
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
    
    def get_client(self):
        """Public method to get the underlying client."""
        return self._get_client()


__all__ = ["AzureOpenAIClient"]
