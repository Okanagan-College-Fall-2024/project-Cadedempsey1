import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load multiple case studies from JSON
with open("case_study.json", "r") as f:
    case_data = json.load(f)

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

# Iterate over case studies
for case_study in case_data["case_studies"]:
    print(f"\n### Case Study: {case_study['title']} ###")
    print(case_study["content"])
    print("\nQuestions and Answers:")

    conversation = [{"role": "system", "content": "You are a helpful AI assistant."}]
    conversation.append({"role": "user", "content": f"Case Study: {case_study['content']}"})

    for question in case_study["questions"]:
        conversation.append({"role": "user", "content": question})
        output = pipe(conversation, **generation_args)
        answer = output[0]["generated_text"]
        print(f"Q: {question}\nA: {answer}\n")
        conversation.append({"role": "assistant", "content": answer})
