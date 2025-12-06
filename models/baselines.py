import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'satisfaction.csv'))
results_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "results"))
print(data_path)
data = pd.read_csv(data_path)

demographic_cols = [
    'Timestamp',
    '1. Your Gender',
    '2. Your Age',
    '3. Are you currently....?',
    '4. What is your annual income?'
]

heldout_cols = [
    '20. Will you continue buying at Starbucks?',
    '7. How much time do you normally  spend during your visit?',
    '5. How often do you visit Starbucks?'
]

heldin_cols = [
    '9. Do you have Starbucks membership card?',
    '13. How would you rate the price range at Starbucks?',
    '12. How would you rate the quality of Starbucks compared to other brands (Coffee Bean, Old Town White Coffee..) to be:',
    "8. The nearest Starbucks's outlet to you is...?"
]


df_demo = data[demographic_cols]
df_heldin = data[heldin_cols]
df_heldout = data[heldout_cols]

heldin_props_dict = {}

for col in df_heldin.columns:
    s = (
        df_heldin[col]
        .astype("string")
        .value_counts(normalize=True, dropna=True)
    )
    heldin_props_dict[col] = s

heldout_props_dict = {}

for col in df_heldout.columns:
    s = (
        df_heldout[col]
        .astype("string")
        .value_counts(normalize=True, dropna=True)
    )
    heldout_props_dict[col] = s

def random_guess_metrics(df_heldout, heldout_props_dict, n_reps=1, seed=None):
    """
    For each held-out question, randomly guess answers according to the
    empirical response distribution (heldout_props_dict), and compute
    accuracy, macro-precision, and macro-recall.

    Parameters
    ----------
    df_heldout : pd.DataFrame
        DataFrame with held-out question responses (one column per question).
    heldout_props_dict : dict
        Dict mapping {question -> Series of response probabilities}.
        Typically built via value_counts(normalize=True).
    n_reps : int
        Number of repeated random-guess runs to average over.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Index = question, columns = ['accuracy', 'precision_macro', 'recall_macro'].
    """
    rng = np.random.default_rng(seed)
    results = {}

    for col in df_heldout.columns:
        # true labels as strings; drop missing
        y_true = df_heldout[col].astype("string").to_numpy()
        mask = y_true != "<NA>"
        y_true = y_true[mask]

        # class probabilities for this question
        probs_series = heldout_props_dict[col]
        classes = probs_series.index.to_numpy()
        # p = probs_series.to_numpy()
        p = probs_series.to_numpy(dtype=float)

        accs = []
        precs = []
        recs = []

        for _ in range(n_reps):
            y_pred = rng.choice(classes, size=len(y_true), p=p)

            accs.append(accuracy_score(y_true, y_pred))
            precs.append(precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ))
            recs.append(recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ))

        results[col] = {
            "accuracy": np.mean(accs),
            "precision_macro": np.mean(precs),
            "recall_macro": np.mean(recs),
        }

    return pd.DataFrame(results).T

metrics_random = random_guess_metrics(df_heldout, heldout_props_dict, n_reps=100, seed=42)
random_path = os.path.join(results_dir, "random_dataframe.csv")
metrics_random.to_csv(random_path, index=True)
print(metrics_random)

import os
import json
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import accuracy_score, precision_score, recall_score

import ollama
from ollama import chat


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    data_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "data", "satisfaction.csv")
    )
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    return data


# ------------------------------------------------------------------
# 2. Column definitions
# ------------------------------------------------------------------

DEMOGRAPHIC_COLS = [
    "1. Your Gender",
    "2. Your Age",
    "3. Are you currently....?",
    "4. What is your annual income?",
]

HELDIN_COLS = [
    "9. Do you have Starbucks membership card?",
    "13. How would you rate the price range at Starbucks?",
    "12. How would you rate the quality of Starbucks compared to other brands (Coffee Bean, Old Town White Coffee..) to be:",
    "8. The nearest Starbucks's outlet to you is...?",
]

HELDOUT_COLS = [
    "20. Will you continue buying at Starbucks?",
    "7. How much time do you normally  spend during your visit?",
    "5. How often do you visit Starbucks?",
]


# ------------------------------------------------------------------
# 3. Pydantic model for structured GPT output
# ------------------------------------------------------------------

class SurveyPrediction(BaseModel):
    answer: str = Field(description="One of the allowed answer choices.")
    explanation: str


# ------------------------------------------------------------------
# 4. GPT/Ollama prediction function for a single question/row
# ------------------------------------------------------------------

def predict_heldout_with_ollama(
    model_name: str,
    context_responses: Dict[str, str],
    heldout_question: str,
    answer_choices: List[str],
) -> SurveyPrediction:
    """
    Use an Ollama model to predict a held-out survey answer.

    Parameters
    ----------
    model_name : str
        Name of the Ollama model, e.g. "gpt-oss:20b".
    context_responses : dict
        Mapping {question: answer} for demographics + held-in questions.
    heldout_question : str
        The question we want to predict.
    answer_choices : list of str
        Allowed categorical responses for this question.

    Returns
    -------
    SurveyPrediction
        Pydantic object with the chosen answer (one of answer_choices)
        and a free-text explanation.
    """

    # JSON schema with ENUM constraint on answer
    ollama_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": answer_choices,
            },
            "explanation": {
                "type": "string",
            },
        },
        "required": ["answer", "explanation"],
    }

    # Build prompt
    prompt = (
        "You are predicting the answer to a multiple-choice survey question.\n\n"
        "Here are this person's previous answers (demographics + held-in questions):\n"
        f"{json.dumps(context_responses, indent=2)}\n\n"
        "Now consider this new question:\n"
        f"Question: {heldout_question}\n\n"
        "Here are the ONLY allowed answer choices for this question:\n"
        f"{json.dumps(answer_choices, indent=2)}\n\n"
        "Based on the previous answers, choose the single most likely answer.\n"
        'You must return a JSON object with fields:\n'
        '{ "answer": <one of the allowed choices>, "explanation": <string> }.\n'
        "Do not return probabilities or any extra fields.\n"
    )

    response = chat(
        model=model_name,
        format=ollama_schema,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    # Validate & coerce into our Pydantic model
    try:
        pred = SurveyPrediction.model_validate_json(response.message.content)
    except ValidationError as e:
        print("Validation error, raw response:")
        print(response.message.content)
        raise e

    # Extra safety check: ensure answer is one of the allowed choices
    if pred.answer not in answer_choices:
        raise ValueError(
            f"Model returned invalid answer: {pred.answer!r}. "
            f"Allowed choices: {answer_choices}"
        )

    return pred


# ------------------------------------------------------------------
# 5. Build answer-choice dict from empirical data
# ------------------------------------------------------------------

def build_answer_choice_dict(df_heldout: pd.DataFrame) -> Dict[str, List[str]]:
    """
    For each held-out question, get the list of observed answer choices
    from the data (unique values, as strings).
    """
    heldout_props_dict = {}
    for col in df_heldout.columns:
        s = (
            df_heldout[col]
            .astype("string")
            .value_counts(normalize=True, dropna=True)
        )
        heldout_props_dict[col] = s

    # Convert Series index to a plain list of strings
    answer_choice_dict = {
        col: list(series.index.astype(str))
        for col, series in heldout_props_dict.items()
    }
    return answer_choice_dict


# ------------------------------------------------------------------
# 6. Loop over respondents and questions, collect predictions
# ------------------------------------------------------------------

def run_gpt_predictions(
    data: pd.DataFrame,
    model_name: str = "gpt-oss:20b",
) -> pd.DataFrame:
    """
    For every respondent and every held-out question, call Ollama
    to predict the answer. Then compute accuracy, macro-precision,
    and macro-recall per question.

    Returns
    -------
    pd.DataFrame
        Index = question, columns = ['accuracy', 'precision_macro', 'recall_macro'].
    """

    # Sub-dataframes
    df_heldout = data[HELDOUT_COLS]

    # Build answer choices per question from empirical data
    answer_choice_dict = build_answer_choice_dict(df_heldout)

    # Storage for true and predicted answers per question
    y_true_dict: Dict[str, List[str]] = {col: [] for col in HELDOUT_COLS}
    y_pred_dict: Dict[str, List[str]] = {col: [] for col in HELDOUT_COLS}

        # Loop over respondents *with progress bar*
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="GPT predicting", ncols=100):
        # Build context responses for this respondent
        context_cols = DEMOGRAPHIC_COLS + HELDIN_COLS
        context_responses = (
            row[context_cols]
            .astype("string")
            .replace("<NA>", "")
            .to_dict()
        )

        # Predict each held-out question for this respondent
        for q in HELDOUT_COLS:
            true_answer = str(row[q])

            # Skip missing true labels
            if true_answer == "nan" or true_answer == "<NA>":
                continue

            answer_choices = answer_choice_dict[q]

            try:
                pred = predict_heldout_with_ollama(
                    model_name=model_name,
                    context_responses=context_responses,
                    heldout_question=q,
                    answer_choices=answer_choices,
                )
            except Exception as e:
                print(f"Skipping idx={idx}, question={q} due to error: {e}")
                continue

            y_true_dict[q].append(true_answer)
            y_pred_dict[q].append(pred.answer)


    # Compute metrics per question
    rows = []
    for q in HELDOUT_COLS:
        y_true = np.array(y_true_dict[q])
        y_pred = np.array(y_pred_dict[q])

        if len(y_true) == 0:
            print(f"No valid examples for question: {q}")
            continue

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        rec = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        rows.append(
            {
                "question": q,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "n": len(y_true),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("question")
    return metrics_df


# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()

    # Optional: inspect columns
    print("Columns in data:")
    print(list(data.columns))
    model_name =  "llama3.1:8b" #"deepseek-r1:8b" # "gpt-oss:20b"
    save_name = f"{str(model_name)}_dataframe.csv".replace(":", "_")
    # Run GPT-based predictions for held-out questions
    metrics_df = run_gpt_predictions(data, model_name=model_name)

    print("\nGPT-based prediction metrics for held-out questions:")
    print(metrics_df)

    llm_path = os.path.join(results_dir, save_name)
    metrics_df.to_csv(llm_path, index=True)
    
