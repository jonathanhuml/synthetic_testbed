"""Compare our agent results with the reference results."""
import pandas as pd

# Load both CSVs
our_results = pd.read_csv('agent_testing/agent_predictions_dataframe.csv')
ref_results = pd.read_csv('agent_testing/llama3.1_8b_dataframe.csv')

print("="*80)
print("RESULTS COMPARISON: Our Generative Agents vs Reference (llama3.1_8b)")
print("="*80)
print()

# Compare each question
for q in our_results['question']:
    our_row = our_results[our_results['question'] == q].iloc[0]
    ref_row = ref_results[ref_results['question'] == q].iloc[0]
    
    print(f"Question: {q}")
    print("-"*80)
    
    acc_diff = our_row['accuracy'] - ref_row['accuracy']
    prec_diff = our_row['precision_macro'] - ref_row['precision_macro']
    rec_diff = our_row['recall_macro'] - ref_row['recall_macro']
    
    print(f"  Accuracy:   Our={our_row['accuracy']:.4f}  |  Ref={ref_row['accuracy']:.4f}  |  Diff={acc_diff:+.4f} {'✓' if acc_diff > 0 else '✗'}")
    print(f"  Precision:  Our={our_row['precision_macro']:.4f}  |  Ref={ref_row['precision_macro']:.4f}  |  Diff={prec_diff:+.4f} {'✓' if prec_diff > 0 else '✗'}")
    print(f"  Recall:     Our={our_row['recall_macro']:.4f}  |  Ref={ref_row['recall_macro']:.4f}  |  Diff={rec_diff:+.4f} {'✓' if rec_diff > 0 else '✗'}")
    print(f"  N:          Our={int(our_row['n'])}  |  Ref={int(ref_row['n'])}")
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print("✓ = Our agents performed better")
print("✗ = Reference performed better")
print()

