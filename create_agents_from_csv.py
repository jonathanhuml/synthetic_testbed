import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
from genagents.genagents import GenerativeAgent
from simulation_engine.global_methods import create_folder_if_not_there

# Define HELDOUT_COLS directly (matching baselines.py)
HELDOUT_COLS = [
    "20. Will you continue buying at Starbucks?",
    "7. How much time do you normally  spend during your visit?",
    "5. How often do you visit Starbucks?",
]


def clean_column_name(col_name):
    """
    Remove leading numbers and periods from column names.
    
    Parameters:
      col_name: Original column name (e.g., "1. Your Gender")
    Returns:
      str: Cleaned column name (e.g., "Your Gender")
    """
    cleaned = re.sub(r'^\d+\.\s*', '', col_name)
    return cleaned.strip()


def create_agents_from_csv(csv_path, output_dir="agents/starbucks_agents", 
                           mask_fields=None, reflection_anchors=None):
    """
    Create multiple agents from CSV where each row becomes one agent.
    
    Structure:
    - First 4 columns (after Timestamp) → scratchpad (demographics)
    - Rest of columns → memories (except masked fields)
    - Masked fields → excluded from memories (for prediction)
    - After memories are added, reflections are generated based on anchors
    
    Parameters:
      csv_path: Path to satisfaction.csv
      output_dir: Directory to save agents
      mask_fields: List of column names to mask (defaults to cleaned HELDOUT_COLS)
      reflection_anchors: List of anchor topics for reflection (defaults to Starbucks-related topics)
    Returns:
      List of agent folder paths
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Clean column names
    column_mapping = {col: clean_column_name(col) for col in df.columns}
    df_cleaned = df.rename(columns=column_mapping)
    
    # First 4 columns are demographics (scratchpad)
    # Skip Timestamp if it's the first column
    if 'Timestamp' in df_cleaned.columns:
        demographic_cols = list(df_cleaned.columns[1:5])  # Skip Timestamp, take next 4
    else:
        demographic_cols = list(df_cleaned.columns[:4])  # First 4 columns
    
    # Masked fields (default to HELDOUT_COLS, but cleaned)
    if mask_fields is None:
        mask_fields_original = HELDOUT_COLS
        mask_fields = [clean_column_name(col) for col in mask_fields_original]
    
    # All other columns (except demographics and masked) go to memories
    memory_cols = [col for col in df_cleaned.columns 
                   if col not in demographic_cols 
                   and col not in mask_fields
                   and col != 'Timestamp']
    
    # Default reflection anchors for Starbucks survey data
    if reflection_anchors is None:
        reflection_anchors = [
            "Starbucks experience and preferences",
            "coffee and beverage habits",
            "customer satisfaction and loyalty"
        ]
    
    print(f"Demographics (scratchpad): {demographic_cols}")
    print(f"Memory fields: {len(memory_cols)} columns")
    print(f"Masked fields (for prediction): {mask_fields}")
    print(f"Reflection anchors: {reflection_anchors}")
    
    # Create output directory
    create_folder_if_not_there(output_dir)
    
    agent_paths = []
    
    # Use tqdm for progress bar
    for idx, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned), 
                         desc="Creating agents", unit="agent"):
        # Create agent
        agent = GenerativeAgent()
        
        # Convert row to dict
        # Keep all values - NaN will be converted to "not available" in memory
        row_dict = row.to_dict()
        
        # 1. Put first 4 columns in scratchpad
        # For demographics, convert NaN to "not available"
        demographics = {}
        for k in demographic_cols:
            if k in row_dict:
                val = row_dict[k]
                if pd.isna(val):
                    demographics[k] = "not available"
                else:
                    demographics[k] = val
        agent.update_scratch(demographics)
        
        # 2. Put remaining columns (except masked) in memories
        # Include all memory columns, even if NaN (will be "not available")
        memory_data = {k: v for k, v in row_dict.items() 
                      if k in memory_cols}
        
        if memory_data:
            memory_text = agent._convert_row_to_memory_text(memory_data)
            agent.remember(memory_text, time_step=1)
        
        # 3. Generate reflections on the agent's memories
        # Reflections help the agent form insights about their preferences and behaviors
        # Only reflect if agent has memories to reflect on
        if memory_data and len(agent.memory_stream.seq_nodes) > 0:
            for anchor in reflection_anchors:
                # Use reflection_count=3 for initial creation (fewer than default 5)
                # Use time_step=2 to indicate reflections come after observations (time_step=1)
                # Use smaller retrieval_count since we have limited memories at creation time
                # Retries are handled internally by chat_safe_generate (3 retries)
                # If it fails after retries, exception will be raised and agent creation will fail
                agent.reflect(anchor=anchor, time_step=2, reflection_count=3, retrieval_count=50)
        
        # 4. Masked fields are NOT added to memories (for prediction later)
        
        # Save agent
        agent_id = f"agent_{idx:04d}"
        agent_folder = Path(output_dir) / agent_id
        agent.save(str(agent_folder))
        agent_paths.append(str(agent_folder))
    
    print(f"\nCreated {len(agent_paths)} agents in {output_dir}")
    return agent_paths


def predict_masked_fields_for_agent(agent_folder, mask_fields=None):
    """
    Predict masked fields for a single agent.
    Uses agent's demographics (scratchpad) + memories to predict masked fields.
    
    Parameters:
      agent_folder: Path to agent folder
      mask_fields: List of field names to predict (defaults to cleaned HELDOUT_COLS)
    Returns:
      Dict of predictions: {field_name: predicted_value}
    """
    if mask_fields is None:
        mask_fields_original = HELDOUT_COLS
        mask_fields = [clean_column_name(col) for col in mask_fields_original]
    
    agent = GenerativeAgent(agent_folder)
    predictions = {}
    
    # Answer choices for masked fields (based on observed values in CSV)
    answer_choices = {
        "Will you continue buying at Starbucks?": ["Yes", "No"],
        "How much time do you normally  spend during your visit?": [
            "Below 30 minutes",
            "Between 30 minutes to 1 hour",
            "Between 1 hour to 2 hours",
            "Between 2 hours to 3 hours",
            "More than 3 hours"
        ],
        "How often do you visit Starbucks?": [
            "Rarely", "Monthly", "Weekly", "Daily", "Never"
        ]
    }
    
    for field_name in mask_fields:
        options = answer_choices.get(field_name, ["Yes", "No"])
        
        question = f"What is {field_name}?"
        questions = {question: options}
        
        try:
            response = agent.categorical_resp(questions)
            if response and "responses" in response and len(response["responses"]) > 0:
                predictions[field_name] = response["responses"][0]
            else:
                predictions[field_name] = None
        except Exception as e:
            print(f"Error predicting {field_name}: {e}")
            predictions[field_name] = None
    
    return predictions


if __name__ == "__main__":
    # Create all agents
    agent_paths = create_agents_from_csv("data/satisfaction.csv")
    
    # Example: predict for first agent
    if agent_paths:
        print("\nPredicting masked fields for first agent...")
        predictions = predict_masked_fields_for_agent(agent_paths[0])
        print(json.dumps(predictions, indent=2))

