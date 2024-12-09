import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load JSON data
with open("data.json", "r") as f:
    data = json.load(f)

# Define a helper function to fetch answers from JSON
def fetch_answer_from_json(question, data):
    if "bananas" in question and "dragonfruits" in question:
        return "\n".join(data.get("bananas_and_dragonfruits", []))
    if "2x + 3 = 7" in question:
        return data.get("equations", {}).get("2x + 3 = 7", "I don't know the answer.")
    return "I couldn't find an answer in the JSON file."

torch.random.manual_seed(0)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Simulated conversation
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
]

# Process user question
question = messages[-1]["content"]
answer_from_json = fetch_answer_from_json(question, data)

# Append the fetched answer to the model's context
if answer_from_json:
    messages.append({"role": "assistant", "content": answer_from_json})

# Generate response using the pipeline
output = pipe(messages, **generation_args)
print(output[0]["generated_text"])
