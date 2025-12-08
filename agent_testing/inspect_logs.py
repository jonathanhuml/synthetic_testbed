"""
Script to inspect the detailed prediction logs.
Shows examples of predictions, errors, and formatting issues.
"""
import pandas as pd
import json
from pathlib import Path

def inspect_detailed_logs(log_csv_path="agent_testing/agent_predictions_detailed_log.csv"):
    """Inspect the detailed log CSV to verify parsing and formatting."""
    
    if not Path(log_csv_path).exists():
        print(f"Log file not found: {log_csv_path}")
        print("Run test_agents.py first to generate the log file.")
        return
    
    df = pd.read_csv(log_csv_path)
    
    print("="*80)
    print("DETAILED PREDICTION LOG INSPECTION")
    print("="*80)
    print(f"Total log entries: {len(df)}")
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-"*80)
    print(f"Successful predictions: {df['is_valid'].sum()}")
    print(f"Failed predictions: {(~df['is_valid']).sum()}")
    print(f"Matches with ground truth: {df['match'].sum()}")
    print()
    
    # Show examples of successful predictions
    print("="*80)
    print("EXAMPLES: SUCCESSFUL PREDICTIONS")
    print("="*80)
    successful = df[df['is_valid'] == True].head(3)
    for idx, row in successful.iterrows():
        print(f"\nAgent: {row['agent_id']} | Question: {row['question']}")
        print(f"Ground Truth: {row['ground_truth']}")
        print(f"Predicted: {row['extracted_prediction']}")
        print(f"Match: {'✓' if row['match'] else '✗'}")
        if pd.notna(row['reasoning']):
            print(f"Reasoning: {row['reasoning'][:200]}...")
        print(f"Parsed Response Structure:")
        try:
            parsed = json.loads(row['parsed_response'])
            print(json.dumps(parsed, indent=2)[:500] + "...")
        except:
            print(row['parsed_response'][:500])
        print("-"*80)
    
    # Show examples of failed predictions
    print("\n" + "="*80)
    print("EXAMPLES: FAILED PREDICTIONS")
    print("="*80)
    failed = df[df['is_valid'] == False].head(5)
    if len(failed) > 0:
        for idx, row in failed.iterrows():
            print(f"\nAgent: {row['agent_id']} | Question: {row['question']}")
            print(f"Ground Truth: {row['ground_truth']}")
            print(f"Error: {row['error']}")
            if pd.notna(row['extracted_prediction']):
                print(f"Extracted (but invalid): {row['extracted_prediction']}")
            print(f"Raw Response Info: {str(row['raw_response'])[:300] if pd.notna(row['raw_response']) else 'None'}...")
            print(f"Parsed Response:")
            try:
                parsed = json.loads(row['parsed_response'])
                print(json.dumps(parsed, indent=2)[:500] + "...")
            except:
                print(str(row['parsed_response'])[:500])
            print("-"*80)
    else:
        print("No failed predictions!")
    
    # Show examples of invalid answers (not in valid choices)
    print("\n" + "="*80)
    print("EXAMPLES: INVALID ANSWERS (not in valid choices)")
    print("="*80)
    invalid = df[(df['is_valid'] == False) & (df['extracted_prediction'].notna())]
    if len(invalid) > 0:
        for idx, row in invalid.head(3).iterrows():
            print(f"\nAgent: {row['agent_id']} | Question: {row['question']}")
            print(f"Ground Truth: {row['ground_truth']}")
            print(f"Extracted Prediction: {row['extracted_prediction']}")
            print(f"Valid Choices: {row['answer_choices']}")
            print(f"Error: {row['error']}")
            print("-"*80)
    else:
        print("No invalid answers found!")
    
    # Question-by-question breakdown
    print("\n" + "="*80)
    print("QUESTION-BY-QUESTION BREAKDOWN")
    print("="*80)
    for question in df['question'].unique():
        q_df = df[df['question'] == question]
        print(f"\n{question}")
        print(f"  Total: {len(q_df)}")
        print(f"  Valid: {q_df['is_valid'].sum()}")
        print(f"  Invalid: {(~q_df['is_valid']).sum()}")
        print(f"  Matches: {q_df['match'].sum()}")
        if q_df['is_valid'].sum() > 0:
            accuracy = q_df['match'].sum() / q_df['is_valid'].sum()
            print(f"  Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    inspect_detailed_logs()

