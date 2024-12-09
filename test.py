import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load case study and questions from JSON
with open("data.json", "r") as f:
    case_data = json.load(f)

case_study = case_data["case_study"]
questions = case_data["questions"]

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

# Format the input for the model
conversation = [{"role": "system", "content": "You are a helpful AI assistant."}]
conversation.append({"role": "user", "content": f"Here is a case study: {case_study}"})

# Generate answers for each question
for question in questions:
    conversation.append({"role": "user", "content": question})
    output = pipe(conversation, **generation_args)
    answer = output[0]["generated_text"]
    print(f"Question: {question}\nAnswer: {answer}\n")
    conversation.append({"role": "assistant", "content": answer})
