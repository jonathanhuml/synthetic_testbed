"""
Test script to evaluate generative agents on held-out questions.

This script:
1. Loads all Starbucks agents from agents/starbucks_agents
2. For each agent, predicts answers to held-out questions using agent.categorical_resp()
3. Compares predictions to ground truth from satisfaction.csv
4. Calculates accuracy, precision_macro, and recall_macro per question
5. Outputs results in CSV format matching the reference format
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime

# Add genagents directory to Python path for imports
# This allows imports to work whether running from project root or genagents directory
genagents_dir = Path(__file__).parent.parent
if str(genagents_dir) not in sys.path:
    sys.path.insert(0, str(genagents_dir))

from genagents.genagents import GenerativeAgent
from genagents.modules.interaction import run_gpt_generate_categorical_resp, _main_agent_desc
from create_agents_from_csv import clean_column_name, HELDOUT_COLS
from simulation_engine.settings import LLM_VERS

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Path to the satisfaction CSV with ground truth
CSV_PATH = "data/satisfaction.csv"

# Path to agents directory
AGENTS_DIR = "agents/starbucks_agents"

# Output CSV path
OUTPUT_CSV = "agent_testing/agent_predictions_dataframe.csv"

# Detailed log CSV path (for inspection)
DETAILED_LOG_CSV = "agent_testing/agent_predictions_detailed_log.csv"

# Answer choices for each held-out question (based on observed values)
ANSWER_CHOICES = {
    "20. Will you continue buying at Starbucks?": ["Yes", "No"],
    "7. How much time do you normally  spend during your visit?": [
        "Below 30 minutes",
        "Between 30 minutes to 1 hour",
        "Between 1 hour to 2 hours",
        "Between 2 hours to 3 hours",
        "More than 3 hours"
    ],
    "5. How often do you visit Starbucks?": [
        "Rarely", "Monthly", "Weekly", "Daily", "Never"
    ]
}


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def load_ground_truth_data(csv_path):
    """
    Load the satisfaction CSV and extract ground truth for held-out questions.
    
    Returns:
        pd.DataFrame: DataFrame with columns for each held-out question
    """
    df = pd.read_csv(csv_path)
    return df


def get_agent_id_from_index(idx):
    """
    Convert agent index to agent folder name.
    
    Parameters:
        idx: Integer index (0-121)
    Returns:
        str: Agent folder name (e.g., "agent_0000")
    """
    return f"agent_{idx:04d}"


def predict_with_agent_single_attempt(agent_folder, question, answer_choices, log_entry=None):
    """
    Single attempt to get a prediction from an agent.
    Internal function used by predict_with_agent which handles retries.
    
    Parameters:
        agent_folder: Path to agent folder
        question: The question to ask
        answer_choices: List of allowed answer choices
        log_entry: Dict to populate with logging information (optional)
    Returns:
        tuple: (predicted_answer, log_info_dict, is_valid) where is_valid is bool
    """
    log_info = {
        "raw_response": None,
        "parsed_response": None,
        "extracted_prediction": None,
        "is_valid": False,
        "error": None,
        "reasoning": None,
        "prompt_used": None
    }
    
    if log_entry is not None:
        log_info.update(log_entry)
    
    try:
        # Load the agent
        agent = GenerativeAgent(agent_folder)
        
        # Format question for categorical response
        # Use the original question text (with number prefix)
        questions = {question: answer_choices}
        
        # Get prediction using lower-level function to capture raw response
        anchor = " ".join(list(questions.keys()))
        agent_desc = _main_agent_desc(agent, anchor)
        
        # Call the function that returns both output and metadata
        output, metadata = run_gpt_generate_categorical_resp(
            agent_desc, questions, "1", LLM_VERS
        )
        
        # Extract raw response from metadata if available
        # metadata is [output, prompt, prompt_input, fail_safe]
        if metadata and len(metadata) >= 2:
            # The raw response before cleanup would be in gpt_request, but we can log what we have
            # The prompt is in metadata[1]
            log_info["prompt_used"] = metadata[1] if metadata[1] else None
        
        # output is the cleaned/parsed response
        response = output
        
        # Log the full response structure
        log_info["parsed_response"] = json.dumps(response, indent=2) if response else "None"
        
        # Try to get raw response - if we can access it from the LLM call
        # For now, we'll note that the raw response is the LLM output before JSON parsing
        # The parsed_response shows what was extracted
        
        # Extract the predicted answer
        if response and "responses" in response and len(response["responses"]) > 0:
            predicted = response["responses"][0]
            log_info["extracted_prediction"] = predicted
            
            # Extract reasoning if available
            if "reasonings" in response and len(response["reasonings"]) > 0:
                log_info["reasoning"] = response["reasonings"][0]
                reasoning_text = response["reasonings"][0]
            else:
                reasoning_text = ""
            
            # FIX: If the LLM returned a single letter (A, B, C, D, E) instead of full text,
            # try to extract the full answer from the reasoning text
            if len(predicted) == 1 and predicted.isalpha() and predicted.upper() in ['A', 'B', 'C', 'D', 'E', 'F']:
                import re
                # Look for pattern like "option B ('Between 30 minutes to 1 hour')" or "option B: 'Between...'"
                # Try multiple patterns
                patterns = [
                    rf"option\s+{predicted.upper()}\s*[:\-\(]\s*['\"]([^'\"]+)['\"]",  # option B ('text')
                    rf"option\s+{predicted.upper()}\s*[:\-\(]\s*([^\)]+)\)",  # option B (text)
                    rf"{predicted.upper()}\s*[:\-\(]\s*['\"]([^'\"]+)['\"]",  # B ('text')
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, reasoning_text, re.IGNORECASE)
                    if match:
                        extracted_full = match.group(1).strip()
                        # Check if the extracted text matches one of the answer choices
                        # Try exact match first
                        if extracted_full in answer_choices:
                            predicted = extracted_full
                            log_info["extracted_prediction"] = predicted
                            log_info["raw_response"] = f"Fixed: LLM returned '{response['responses'][0]}', extracted '{predicted}' from reasoning"
                            break
                        # Try partial match (in case of quotes or extra text)
                        for choice in answer_choices:
                            if extracted_full in choice or choice in extracted_full:
                                predicted = choice
                                log_info["extracted_prediction"] = predicted
                                log_info["raw_response"] = f"Fixed: LLM returned '{response['responses'][0]}', matched '{predicted}' from reasoning"
                                break
                        if predicted in answer_choices:
                            break
            
            # FIX: Strip number prefixes (e.g., "1. Answer" -> "Answer")
            # This handles cases where LLM includes the number from numbered lists
            import re
            predicted_cleaned = re.sub(r'^\d+\.\s*', '', predicted).strip()
            if predicted_cleaned != predicted:
                log_info["raw_response"] = f"Stripped number prefix: '{predicted}' -> '{predicted_cleaned}'"
                predicted = predicted_cleaned
                log_info["extracted_prediction"] = predicted
            
            # Ensure the prediction is one of the valid choices
            if predicted in answer_choices:
                log_info["is_valid"] = True
                return predicted, log_info, True
            else:
                # Try to find a match by removing common prefixes/suffixes
                for choice in answer_choices:
                    # Check if predicted contains the choice or vice versa
                    if predicted in choice or choice in predicted:
                        # Prefer exact match or choice that contains predicted
                        if choice in predicted or len(choice) <= len(predicted):
                            predicted = choice
                            log_info["extracted_prediction"] = predicted
                            log_info["raw_response"] = f"Matched partial: '{response['responses'][0]}' -> '{predicted}'"
                            if predicted in answer_choices:
                                log_info["is_valid"] = True
                                return predicted, log_info, True
                
                log_info["error"] = f"Invalid answer '{predicted}' not in valid choices: {answer_choices}"
                log_info["is_valid"] = False
                # Still log the raw response for debugging
                if not log_info.get("raw_response"):
                    log_info["raw_response"] = f"LLM returned: {predicted} (not in valid choices)"
                return None, log_info, False
        else:
            log_info["error"] = "Empty response or no 'responses' field"
            log_info["is_valid"] = False
            if response:
                log_info["raw_response"] = f"Response structure: {json.dumps(response)}"
            return None, log_info, False
            
    except Exception as e:
        import traceback
        error_msg = f"Exception: {str(e)}"
        error_trace = traceback.format_exc()
        log_info["error"] = error_msg
        log_info["raw_response"] = f"Error traceback: {error_trace}"
        log_info["is_valid"] = False
        return None, log_info, False


def predict_with_agent(agent_folder, question, answer_choices, log_entry=None, max_retries=3):
    """
    Use a generative agent to predict the answer to a question with retry logic.
    Retries up to max_retries times if the response is invalid.
    
    Parameters:
        agent_folder: Path to agent folder
        question: The question to ask
        answer_choices: List of allowed answer choices
        log_entry: Dict to populate with logging information (optional)
        max_retries: Maximum number of retry attempts (default: 3)
    Returns:
        tuple: (predicted_answer, log_info_dict) where predicted_answer is str or None
    """
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        predicted, log_info, is_valid = predict_with_agent_single_attempt(
            agent_folder, question, answer_choices, log_entry
        )
        
        if is_valid:
            # Success! Add retry info if we retried
            if retry_count > 0:
                log_info["raw_response"] = f"{log_info.get('raw_response', '')} [Retried {retry_count} time(s)]"
            return predicted, log_info
        
        # Invalid response - prepare for retry
        last_error = log_info.get("error", "Unknown error")
        retry_count += 1
        
        if retry_count <= max_retries:
            # Wait a bit before retrying (exponential backoff)
            wait_time = min(2 ** (retry_count - 1), 5)  # Max 5 seconds
            time.sleep(wait_time)
            # Update log entry with retry info
            if log_entry is not None:
                log_entry["retry_attempt"] = retry_count
        else:
            # Max retries reached
            log_info["error"] = f"Failed after {max_retries} retries. Last error: {last_error}"
            log_info["raw_response"] = f"{log_info.get('raw_response', '')} [Failed after {max_retries} retries]"
            return None, log_info
    
    # Should not reach here, but just in case
    log_info["error"] = f"Failed after {max_retries} retries. Last error: {last_error}"
    return None, log_info


# ------------------------------------------------------------------
# Main Testing Function
# ------------------------------------------------------------------

def run_agent_predictions(csv_path, agents_dir, output_csv, detailed_log_csv=None):
    """
    Run predictions for all agents on all held-out questions and compute metrics.
    
    Parameters:
        csv_path: Path to satisfaction.csv with ground truth
        agents_dir: Directory containing agent folders
        output_csv: Path to save results CSV
        detailed_log_csv: Path to save detailed log CSV (optional)
    Returns:
        pd.DataFrame: Metrics per question
    """
    print("="*80)
    print("GENERATIVE AGENTS PREDICTION TEST")
    print("="*80)
    print(f"Loading ground truth from: {csv_path}")
    print(f"Loading agents from: {agents_dir}")
    if detailed_log_csv:
        print(f"Detailed log will be saved to: {detailed_log_csv}")
    print()
    
    # Load ground truth data
    df_truth = load_ground_truth_data(csv_path)
    n_agents = len(df_truth)
    print(f"Found {n_agents} respondents in CSV")
    
    # Verify agents directory exists
    agents_path = Path(agents_dir)
    if not agents_path.exists():
        raise FileNotFoundError(f"Agents directory not found: {agents_dir}")
    
    # Storage for true and predicted answers per question
    y_true_dict = {col: [] for col in HELDOUT_COLS}
    y_pred_dict = {col: [] for col in HELDOUT_COLS}
    
    # Track which agents/questions we successfully processed
    successful_predictions = {col: 0 for col in HELDOUT_COLS}
    
    # Detailed log entries for inspection
    detailed_logs = []
    
    print("\nRunning predictions for all agents...")
    print("-" * 80)
    
    # Loop over all agents (rows in CSV)
    for idx in tqdm(range(n_agents), desc="Processing agents", ncols=100):
        agent_id = get_agent_id_from_index(idx)
        agent_folder = agents_path / agent_id
        
        # Skip if agent doesn't exist
        if not agent_folder.exists():
            print(f"Warning: Agent {agent_id} not found, skipping...")
            continue
        
        # Get ground truth for this agent (row in CSV)
        row = df_truth.iloc[idx]
        
        # Predict each held-out question for this agent
        for question in HELDOUT_COLS:
            # Get ground truth answer
            true_answer = str(row[question])
            
            # Skip missing ground truth
            if pd.isna(row[question]) or true_answer == "nan" or true_answer == "<NA>":
                continue
            
            # Get answer choices for this question
            answer_choices = ANSWER_CHOICES[question]
            
            # Create log entry
            log_entry = {
                "agent_id": agent_id,
                "agent_index": idx,
                "question": question,
                "ground_truth": true_answer,
                "answer_choices": json.dumps(answer_choices),
                "timestamp": datetime.now().isoformat()
            }
            
            # Get prediction from agent with logging
            predicted, log_info = predict_with_agent(agent_folder, question, answer_choices, log_entry)
            
            # Merge log info
            log_entry.update(log_info)
            log_entry["match"] = (predicted == true_answer) if predicted is not None else False
            
            # Add to detailed logs
            detailed_logs.append(log_entry)
            
            # Store results if prediction succeeded
            if predicted is not None:
                y_true_dict[question].append(true_answer)
                y_pred_dict[question].append(predicted)
                successful_predictions[question] += 1
    
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    for question in HELDOUT_COLS:
        n = successful_predictions[question]
        print(f"{question}: {n} successful predictions")
    print()
    
    # Compute metrics per question
    print("Computing metrics...")
    print("-" * 80)
    
    rows = []
    for question in HELDOUT_COLS:
        y_true = np.array(y_true_dict[question])
        y_pred = np.array(y_pred_dict[question])
        
        if len(y_true) == 0:
            print(f"Warning: No valid predictions for question: {question}")
            continue
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        rec = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        
        rows.append({
            "question": question,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "n": len(y_true),
        })
        
        print(f"{question}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision (macro): {prec:.4f}")
        print(f"  Recall (macro): {rec:.4f}")
        print(f"  N: {len(y_true)}")
        print()
    
    # Create results DataFrame
    metrics_df = pd.DataFrame(rows)
    
    # Save to CSV (matching reference format)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    
    print("="*80)
    print(f"Results saved to: {output_csv}")
    
    # Save detailed log if requested
    if detailed_log_csv and detailed_logs:
        log_df = pd.DataFrame(detailed_logs)
        log_path = Path(detailed_log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)
        print(f"Detailed log saved to: {detailed_log_csv}")
        print(f"  Total log entries: {len(detailed_logs)}")
        print(f"  Successful predictions: {sum(successful_predictions.values())}")
        print(f"  Failed predictions: {len(detailed_logs) - sum(successful_predictions.values())}")
    
    print("="*80)
    
    return metrics_df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure we're in the genagents directory
    script_dir = Path(__file__).parent.parent
    if os.path.basename(os.getcwd()) != "genagents":
        if script_dir.exists() and script_dir.name == "genagents":
            os.chdir(script_dir)
            print(f"Changed working directory to: {os.getcwd()}")
    
    # Run the test with detailed logging
    metrics_df = run_agent_predictions(CSV_PATH, AGENTS_DIR, OUTPUT_CSV, DETAILED_LOG_CSV)
    
    # Display results
    print("\nFINAL RESULTS:")
    print(metrics_df.to_string(index=False))

