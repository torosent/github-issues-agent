"""Configuration module for GitHub Issues AI Agent."""
import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv, dotenv_values


# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Runtime configuration values sourced from environment variables.

    Attributes map directly to required env vars. All are mandatory; absence
    results in a ValueError during loading.
    """
    github_token: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    azure_openai_api_version: str


def load_config() -> Config:
    """Load configuration from process environment and optional .env file.

    Loading order per variable:
      1. Real environment (`os.environ`).
      2. `.env` file key (if present) via `dotenv_values`.

    Returns:
        Config: Populated immutable configuration object.

    Raises:
        ValueError: If any required variable is absent after merging sources.
    """
    # Load .env file if present - first try load_dotenv, then fallback to dotenv_values
    try:
        load_dotenv()
    except (AssertionError, Exception):
        # load_dotenv can fail in some test environments or Python versions
        pass
    
    # If .env exists, also read values directly and merge
    env_values = {}
    if os.path.exists('.env'):
        env_values = dotenv_values('.env')
    
    # Required environment variables
    required_vars = {
        'GITHUB_TOKEN': 'github_token',
        'AZURE_OPENAI_ENDPOINT': 'azure_openai_endpoint',
        'AZURE_OPENAI_API_KEY': 'azure_openai_api_key',
        'AZURE_OPENAI_DEPLOYMENT': 'azure_openai_deployment',
        'AZURE_OPENAI_API_VERSION': 'azure_openai_api_version',
    }
    
    config_values = {}
    missing_vars = []
    
    for env_var, field_name in required_vars.items():
        # Try environment first, then .env file values
        value = os.getenv(env_var) or env_values.get(env_var)
        if not value:
            missing_vars.append(env_var)
        else:
            config_values[field_name] = value
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    logger.info("Configuration loaded successfully")
    return Config(**config_values)
