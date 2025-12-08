import ollama
import time
from typing import List, Union

from simulation_engine.settings import *


# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def gpt_request(prompt: str, 
                model: str = "llama3", 
                max_tokens: int = 1500) -> str:
  """Make a request to Ollama's chat model."""
  try:
    # Initialize Ollama client with base URL from settings
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      options={
        "num_predict": max_tokens,
        "temperature": 0.7
      }
    )
    # Extract content from response - handle both dict and object responses
    if isinstance(response, dict):
      return response.get("message", {}).get("content", "")
    else:
      return getattr(response.message, "content", "")
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "llama3", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500,
                       file_attachment: str = None,
                       file_type: str = None) -> tuple:
  """Generate a response using Ollama models with error handling & retries."""
  # Generate prompt from template
  prompt = generate_prompt(prompt_input, prompt_lib_file)
  
  # Retry logic for robust generation
  for i in range(repeat):
    response = gpt_request(prompt, model=gpt_version, max_tokens=max_tokens)
    if response != "GENERATION ERROR":
      break
    time.sleep(2**i)
  else:
    response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def get_text_embedding(text: str, 
                       model: str = None) -> List[float]:
  """Generate an embedding for the given text using Ollama's API (fully local)."""
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")

  # Use default embedding model from settings if not specified
  if model is None:
    model = OLLAMA_EMBEDDING_MODEL

  text = text.replace("\n", " ").strip()
  
  try:
    # Initialize Ollama client with base URL from settings
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.embeddings(model=model, prompt=text)
    # Extract embedding from response - handle both dict and object responses
    if isinstance(response, dict):
      return response.get("embedding", [])
    else:
      return getattr(response, "embedding", [])
  except Exception as e:
    raise ValueError(f"Failed to generate embedding: {str(e)}")









