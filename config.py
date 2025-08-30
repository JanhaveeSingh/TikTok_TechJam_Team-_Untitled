# LLM Configuration
# These settings control the connection and behavior of the Large Language Model API
# Please fill in your own API information below


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configurationx
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # 'gemini', 'openai', 'deepseek', etc.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-pro")  # e.g., 'gemini-pro', 'gpt-4o', etc.
LLM_MAX_TOKEN = int(os.getenv("LLM_MAX_TOKEN", 1500))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", 500))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/")

# Gemini-specific (if needed)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", LLM_API_KEY)


# LangChain Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "AppAgentX")


# Neo4j Configuration
Neo4j_URI = os.getenv("NEO4J_URI")
Neo4j_AUTH = (
	os.getenv("NEO4J_USERNAME"),
	os.getenv("NEO4J_PASSWORD")
)

# Feature Extractor Configuration
Feature_URI = os.getenv("FEATURE_URI", "http://127.0.0.1:8001")

# Screen Parser Configuration
Omni_URI = os.getenv("OMNI_URI", "http://127.0.0.1:8000")
CLIP_URI = os.getenv("CLIP_URI", "http://127.0.0.1:8002")  # Different port to avoid conflict

# Parser Selection Configuration
PARSER_TYPE = os.getenv("PARSER_TYPE", "auto")  # "omni", "clip", "auto"
PRIMARY_PARSER = os.getenv("PRIMARY_PARSER", "omni")  # Primary choice
FALLBACK_PARSER = os.getenv("FALLBACK_PARSER", "clip")  # Fallback choice

# Vector Storage Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
