# Generative Agents Project - Complete Summary

## Table of Contents

1. [Project Goal](#project-goal)
2. [What This Project Does](#what-this-project-does)
3. [Key Changes Made](#key-changes-made)
4. [Installation & Setup](#installation--setup)
5. [Project Structure](#project-structure)
6. [How to Use](#how-to-use)
7. [All Commands](#all-commands)
8. [Testing & Evaluation](#testing--evaluation)
9. [Results](#results)
10. [Technical Details](#technical-details)

---

## Project Goal

This project is a **local, privacy-preserving implementation** of the Stanford Generative Agents framework. The original project used OpenAI's API to create AI agents that simulate real people based on interview data. This clone has been modified to use **Ollama** (a local LLM framework) instead, making it:

- **Fully local** - No data sent to external APIs
- **Privacy-preserving** - All processing happens on your machine
- **Cost-free** - No API costs
- **Customizable** - Use any Ollama-compatible model

The project demonstrates how to:

1. Create generative agents from survey/CSV data
2. Store agent memories and demographics
3. Query agents with questions and get responses
4. Evaluate agent performance on held-out questions

---

## What This Project Does

### Core Functionality

1. **Agent Creation**: Converts CSV survey data into generative agents

   - Each row in the CSV becomes one agent
   - Demographics (first 4 columns) → stored in agent's "scratchpad"
   - Survey responses → stored as memories
   - Some fields can be masked (held out) for prediction testing

2. **Agent Interaction**: Query agents with questions

   - **Categorical questions**: Multiple choice (e.g., "Yes/No", "Daily/Weekly/Monthly")
   - **Numerical questions**: Scale-based responses (e.g., 1-10 ratings)
   - **Open-ended questions**: Free-form dialogue

3. **Memory System**: Agents have persistent memory

   - Memories are stored with embeddings for semantic search
   - Agents can reflect on their memories
   - Memory influences agent responses

4. **Testing & Evaluation**: Evaluate agent accuracy
   - Test agents on held-out questions
   - Compare predictions to ground truth
   - Calculate accuracy, precision, and recall metrics

---

## Key Changes Made

### 1. **Ollama Integration (Replaced OpenAI)**

**Original Implementation:**

- Used OpenAI API (`openai` Python package)
- Required API keys and internet connection
- Cost per API call
- Data sent to external servers

**New Implementation:**

- Uses Ollama for local LLM inference
- Fully local processing (no internet required after setup)
- No API costs
- Complete data privacy

**Files Modified:**

- `simulation_engine/settings.py`: Changed from OpenAI config to Ollama config
- `simulation_engine/gpt_structure.py`: Replaced OpenAI API calls with Ollama client
- `requirements.txt`: Removed `openai`, added `ollama`

**Key Configuration Changes:**

```python
# OLD (OpenAI):
OPENAI_API_KEY = "YOUR_API_KEY"
LLM_VERS = "gpt-4o-mini"

# NEW (Ollama):
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
LLM_VERS = "llama3.1:latest"
```

### 2. **CSV-Based Agent Creation**

Added `create_agents_from_csv.py` script that:

- Reads survey data from CSV files
- Automatically creates agents from each row
- Handles demographics vs. memories separation
- Supports masking fields for prediction tasks

### 3. **Testing Framework**

Created comprehensive testing suite in `agent_testing/`:

- `test_agents.py`: Main testing script with detailed logging
- `compare_results.py`: Compare results with reference baselines
- `show_conversations.py`: Display agent conversations
- `inspect_logs.py`: Debug and inspect prediction logs

### 4. **Enhanced Memory Conversion**

Added `_convert_row_to_memory_text()` method to `GenerativeAgent` class:

- Converts CSV row data to natural language memories
- Handles NaN values gracefully
- Formats rating questions (1-5 scale) appropriately
- Creates readable memory descriptions

### 5. **Robust Prediction Handling**

Enhanced prediction system with:

- Retry logic for failed predictions
- Answer validation and normalization
- Detailed logging for debugging
- Handles edge cases (single-letter answers, number prefixes, etc.)

---

## Installation & Setup

### Prerequisites

1. **Python 3.7+**

   ```bash
   python --version  # Should be 3.7 or higher
   ```

2. **Ollama Installed and Running**

   ```bash
   # Install Ollama from https://ollama.ai
   # Then pull required models:
   ollama pull llama3.1:latest
   ollama pull nomic-embed-text

   # Verify Ollama is running:
   ollama list
   ```

3. **Python Dependencies**
   ```bash
   cd genagents
   pip install -r requirements.txt
   ```

### Configuration

1. **Create Settings File**

   ```bash
   # Copy example settings
   cp simulation_engine/example-settings.py simulation_engine/settings.py
   ```

2. **Edit `simulation_engine/settings.py`**

   ```python
   # Update these if needed:
   OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
   LLM_VERS = "llama3.1:latest"  # Your preferred model
   OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model
   ```

3. **Verify Setup**
   ```bash
   # Test Ollama connection
   curl http://localhost:11434/api/tags
   ```

---

## Project Structure

```
genagents/
├── genagents/                    # Core agent module
│   ├── genagents.py              # Main GenerativeAgent class
│   └── modules/
│       ├── interaction.py        # Agent interaction methods
│       └── memory_stream.py      # Memory management
│
├── simulation_engine/             # Engine configuration
│   ├── settings.py               # Configuration (CREATE THIS)
│   ├── example-settings.py       # Example config template
│   ├── gpt_structure.py          # Ollama API wrapper
│   ├── global_methods.py         # Utility functions
│   ├── llm_json_parser.py        # JSON response parsing
│   └── prompt_template/          # LLM prompt templates
│
├── agents/                        # Stored agents
│   └── starbucks_agents/         # Pre-created agents (122 agents)
│       └── agent_0000/           # Individual agent folders
│           ├── scratch.json      # Demographics/attributes
│           ├── meta.json         # Agent metadata
│           └── memory_stream/    # Memory data
│               ├── nodes.json    # Memory nodes
│               └── embeddings.json # Memory embeddings
│
├── data/                          # Input data
│   ├── satisfaction.csv          # Survey data (122 respondents)
│   └── llama3.1_8b_dataframe.csv # Reference results
│
├── results/                       # Output files
│   ├── agent_predictions_dataframe.csv      # Metrics summary
│   └── agent_predictions_detailed_log.csv   # Detailed logs
│
├── agent_testing/                 # Testing scripts
│   ├── test_agents.py            # Main testing script
│   ├── compare_results.py        # Compare with baseline
│   ├── show_conversations.py     # Display conversations
│   └── inspect_logs.py           # Debug logs
│
├── create_agents_from_csv.py     # Agent creation script
├── requirements.txt              # Python dependencies
├── README.md                     # Original project README
└── summary.md                    # This file
```

---

## How to Use

### Step 1: Create Agents from CSV

Convert survey data into agents:

```bash
cd genagents
python create_agents_from_csv.py
```

**What it does:**

- Reads `data/satisfaction.csv`
- Creates 122 agents in `agents/starbucks_agents/`
- First 4 columns → demographics (scratchpad)
- Remaining columns → memories
- Masks 3 held-out questions for testing

**Output:**

```
Created 122 agents in agents/starbucks_agents/
```

### Step 2: Test Agents

Run predictions on held-out questions:

```bash
python agent_testing/test_agents.py
```

**What it does:**

- Loads all agents
- Asks each agent 3 held-out questions
- Compares predictions to ground truth
- Calculates accuracy, precision, recall
- Saves results to `results/`

**Output:**

- `results/agent_predictions_dataframe.csv` - Summary metrics
- `results/agent_predictions_detailed_log.csv` - Detailed logs

### Step 3: View Results

**Compare with baseline:**

```bash
python agent_testing/compare_results.py
```

**View conversations:**

```bash
python agent_testing/show_conversations.py
```

**Inspect logs:**

```bash
python agent_testing/inspect_logs.py
```

### Step 4: Interact with Individual Agents

```python
from genagents.genagents import GenerativeAgent

# Load an agent
agent = GenerativeAgent(agent_folder="agents/starbucks_agents/agent_0000")

# Ask categorical question
questions = {
    "Do you enjoy coffee?": ["Yes", "No", "Sometimes"]
}
response = agent.categorical_resp(questions)
print(response["responses"])

# Ask numerical question
questions = {
    "Rate your satisfaction (1-10):": [1, 10]
}
response = agent.numerical_resp(questions, float_resp=False)
print(response["responses"])

# Open-ended dialogue
dialogue = [
    ("Interviewer", "Tell me about your coffee preferences."),
]
response = agent.utterance(dialogue)
print(response)
```

---

## All Commands

### Setup Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Setup Ollama (if not installed)
# Download from https://ollama.ai
ollama pull llama3.1:latest
ollama pull nomic-embed-text

# Create settings file
cp simulation_engine/example-settings.py simulation_engine/settings.py
# Edit settings.py with your preferences
```

### Agent Creation

```bash
# Create agents from CSV
python create_agents_from_csv.py

# Create agents with custom settings
python -c "
from create_agents_from_csv import create_agents_from_csv
create_agents_from_csv(
    csv_path='data/satisfaction.csv',
    output_dir='agents/my_agents',
    mask_fields=['Question 1', 'Question 2']
)
"
```

### Testing Commands

```bash
# Run full test suite
python agent_testing/test_agents.py

# Compare results with baseline
python agent_testing/compare_results.py

# Show conversations for first 5 agents
python agent_testing/show_conversations.py

# Inspect detailed logs
python agent_testing/inspect_logs.py
```

### Ollama Management

```bash
# List installed models
ollama list

# Pull a model
ollama pull llama3.1:latest
ollama pull nomic-embed-text

# Test Ollama connection
curl http://localhost:11434/api/tags

# Run a model interactively
ollama run llama3.1:latest
```

### Python Interactive Usage

```python
# In Python REPL or script
from genagents.genagents import GenerativeAgent

# Create new agent
agent = GenerativeAgent()
agent.update_scratch({
    "first_name": "John",
    "age": 30,
    "occupation": "Engineer"
})

# Add memory
agent.remember("Loves coffee and coding", time_step=1)

# Save agent
agent.save("agents/my_agent")

# Load agent
agent = GenerativeAgent(agent_folder="agents/my_agent")

# Query agent
questions = {"Do you like coffee?": ["Yes", "No"]}
response = agent.categorical_resp(questions)
print(response)
```

---

## Testing & Evaluation

### Test Framework

The testing framework evaluates agents on **held-out questions** - questions that were masked during agent creation and not included in the agent's memories.

### Held-Out Questions

1. **"20. Will you continue buying at Starbucks?"**

   - Choices: `["Yes", "No"]`
   - Binary classification

2. **"7. How much time do you normally spend during your visit?"**

   - Choices: `["Below 30 minutes", "Between 30 minutes to 1 hour", "Between 1 hour to 2 hours", "Between 2 hours to 3 hours", "More than 3 hours"]`
   - Multi-class classification

3. **"5. How often do you visit Starbucks?"**
   - Choices: `["Rarely", "Monthly", "Weekly", "Daily", "Never"]`
   - Multi-class classification

### Metrics Calculated

For each question:

- **Accuracy**: Percentage of correct predictions
- **Precision (macro)**: Average precision across all classes
- **Recall (macro)**: Average recall across all classes
- **N**: Number of successful predictions

### Testing Process

1. **Load Ground Truth**: Read actual answers from `data/satisfaction.csv`
2. **Load Agents**: Load all agents from `agents/starbucks_agents/`
3. **Query Each Agent**: Ask each held-out question
4. **Validate Responses**: Ensure answers match valid choices
5. **Compare**: Match predictions to ground truth
6. **Calculate Metrics**: Compute accuracy, precision, recall
7. **Save Results**: Write to CSV files

### Retry Logic

The testing script includes robust retry logic:

- Up to 3 retry attempts per question
- Exponential backoff between retries
- Detailed error logging
- Handles edge cases (single-letter answers, formatting issues)

---

## Results

### Current Performance

Based on testing 122 agents with Ollama (llama3.1:latest):

| Question                                               | Accuracy   | Precision (macro) | Recall (macro) | N   |
| ------------------------------------------------------ | ---------- | ----------------- | -------------- | --- |
| Will you continue buying at Starbucks?                 | **78.69%** | 72.99%            | 56.08%         | 122 |
| How much time do you normally spend during your visit? | **54.92%** | 33.24%            | 33.31%         | 122 |
| How often do you visit Starbucks?                      | **22.95%** | 36.00%            | 26.27%         | 122 |

### Performance Notes

- **Best performance**: Binary question (continue buying) - 78.69% accuracy
- **Moderate performance**: Time spent question - 54.92% accuracy
- **Lower performance**: Visit frequency question - 22.95% accuracy

The visit frequency question shows lower accuracy, which may indicate:

- More nuanced decision-making required
- Need for better prompt engineering
- Model limitations with multi-class classification

### Comparison with Baseline

Use `compare_results.py` to compare with reference results from `llama3.1_8b_dataframe.csv`.

---

## Technical Details

### Agent Architecture

**GenerativeAgent Class:**

- `scratch`: Dictionary storing demographics/attributes
- `memory_stream`: MemoryStream object managing memories
- `id`: Unique identifier (UUID)

**Memory System:**

- Memories stored as nodes with embeddings
- Embeddings enable semantic search
- Memories influence agent responses via retrieval

**Interaction Methods:**

- `categorical_resp()`: Multiple choice questions
- `numerical_resp()`: Scale-based questions
- `utterance()`: Open-ended dialogue

### Ollama Integration

**API Wrapper (`gpt_structure.py`):**

- `gpt_request()`: Direct Ollama API call
- `chat_safe_generate()`: Wrapper with retry logic
- `get_text_embedding()`: Generate embeddings via Ollama

**Configuration:**

- Base URL: `http://localhost:11434` (default)
- Chat model: `llama3.1:latest` (configurable)
- Embedding model: `nomic-embed-text` (768 dimensions)

### Data Flow

1. **CSV → Agent Creation:**

   ```
   CSV Row → Demographics (scratchpad) + Memories → Agent Object → Saved to Disk
   ```

2. **Query → Response:**

   ```
   Question → Agent Description (scratch + relevant memories) → LLM Prompt → Ollama API → Parsed Response
   ```

3. **Memory Storage:**
   ```
   Memory Text → Embedding (Ollama) → Stored in nodes.json + embeddings.json
   ```

### File Formats

**Agent Storage:**

- `scratch.json`: Demographics/attributes (JSON)
- `meta.json`: Agent metadata (JSON)
- `memory_stream/nodes.json`: Memory nodes (JSON array)
- `memory_stream/embeddings.json`: Embedding vectors (JSON dict)

**Results:**

- `agent_predictions_dataframe.csv`: Metrics per question (CSV)
- `agent_predictions_detailed_log.csv`: Full prediction logs (CSV)

### Prompt Templates

Prompts are stored in `simulation_engine/prompt_template/generative_agent/`:

- Templates define how agent descriptions are formatted
- Include scratchpad info and relevant memories
- Structured for consistent LLM responses

---

## Troubleshooting

### Common Issues

1. **Ollama not running:**

   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   # If error, start Ollama service
   ```

2. **Model not found:**

   ```bash
   # Pull required models
   ollama pull llama3.1:latest
   ollama pull nomic-embed-text
   ```

3. **Import errors:**

   ```bash
   # Ensure you're in the genagents directory
   cd genagents
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

4. **Settings file missing:**

   ```bash
   # Copy example settings
   cp simulation_engine/example-settings.py simulation_engine/settings.py
   ```

5. **Low accuracy:**
   - Check Ollama model quality
   - Verify prompt templates
   - Review detailed logs for patterns
   - Try different models (e.g., `llama3.2`, `mistral`)

---

## Future Improvements

Potential enhancements:

- Support for more LLM backends (vLLM, llama.cpp)
- Better prompt engineering for multi-class questions
- Agent fine-tuning capabilities
- Batch processing optimizations
- Web interface for agent interaction
- More sophisticated memory retrieval
- Agent-to-agent interactions

---

## References

- **Original Paper**: Park, J. S., et al. (2024). _Generative Agent Simulations of 1,000 People_
- **Ollama**: https://ollama.ai
- **Original Repository**: https://github.com/joonspk-research/generative_agents

---

## License

This project is licensed under the MIT License (see `LICENSE` file).

---

## Contact & Support

For questions about:

- **Original project**: Contact Joon Sung Park (joonspk@stanford.edu)
- **This clone/modification**: See project repository issues

---

**Last Updated**: Based on current codebase state
**Version**: Local Ollama Implementation
