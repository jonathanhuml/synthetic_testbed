import json
from typing import Literal

import ollama
from ollama import chat
from pydantic import BaseModel, Field

# ---------- Constrained output model ----------

class Person(BaseModel):
    yes_or_no: Literal["Yes", "No"]
    confidence: float = Field(ge=0, le=1) 
    explanation: str

# ---------- Data ----------

personal_responses = {
    "Current occupation": "Fruit salesman",
    "Do you like apples?": "Yes",
    "Do you like bananas?": "Yes",
    "Do you like computers?": "Yes",
}

keys = list(personal_responses.keys())
heldout_question = keys[-1]          # last question
context_keys = keys[:-1]             # all but last
context_responses = {
    k: personal_responses[k] for k in context_keys
}

# ---------- Simple JSON schema for Ollama ----------

ollama_schema = {
    "type": "object",
    "properties": {
        "yes_or_no": {"type": "string"},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["yes_or_no", "confidence", "explanation"],
}

response = chat(
    model="gpt-oss:20b",
    format=ollama_schema,        
    messages=[
        {
            "role": "user",
            "content": (
                "You are predicting the answer to a yes/no question.\n\n"
                "Here are this person's previous answers:\n"
                f"{json.dumps(context_responses, indent=2)}\n\n"
                "Now consider this new question:\n"
                f"Question: {heldout_question}\n\n"
                "Based on the previous answers, predict how this person "
                "would answer the new question.\n\n"
            ),
        }
    ],
)

# ---------- Validate & use the result ----------

person = Person.model_validate_json(response.message.content)

print(person)
print("yes_or_no:", person.yes_or_no)
print("confidence:", person.confidence)
print("explanation:", person.explanation)
