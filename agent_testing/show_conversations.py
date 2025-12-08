"""Show conversations for the first 5 agents - questions asked and answers given."""
import pandas as pd
import json

# Load the detailed log
df = pd.read_csv('agent_testing/agent_predictions_detailed_log.csv')

# Get first 5 agents (agent_0000 to agent_0004)
first_5_agents = ['agent_0000', 'agent_0001', 'agent_0002', 'agent_0003', 'agent_0004']

print("="*80)
print("CONVERSATIONS: FIRST 5 AGENTS")
print("="*80)

for agent_id in first_5_agents:
    agent_data = df[df['agent_id'] == agent_id].sort_values('question')
    
    if len(agent_data) == 0:
        continue
    
    print(f"\n{'='*80}")
    print(f"AGENT: {agent_id}")
    print(f"{'='*80}")
    
    for idx, row in agent_data.iterrows():
        print(f"\nQuestion: {row['question']}")
        print(f"Answer Choices: {row['answer_choices']}")
        print(f"\nAgent's Answer: {row['extracted_prediction']}")
        print(f"Ground Truth: {row['ground_truth']}")
        match = "✓" if row['match'] else "✗"
        print(f"Match: {match}")
        
        if pd.notna(row['reasoning']):
            print(f"\nAgent's Reasoning:")
            print(f"  {row['reasoning'][:300]}...")
        
        print("-" * 80)

print("\n" + "="*80)
print("END OF CONVERSATIONS")
print("="*80)

