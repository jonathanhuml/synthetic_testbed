from pathlib import Path

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding configuration - using Ollama for fully local setup
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions

KEY_OWNER = "NAME"

DEBUG = False

MAX_CHUNK_SIZE = 4

# Default Ollama chat model (e.g., "llama3.1", "llama3", "mistral", "qwen2.5")
LLM_VERS = "llama3.1:latest"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

## Agent storage directory - user can customize this path
AGENTS_DIR = f"{BASE_DIR}/agents" 
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"