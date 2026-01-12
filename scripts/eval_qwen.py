import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import difflib

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "c:/Users/MSI/Desktop/ARENA/checkpoints/grpo_New"  # change this path
data_path = "c:/Users/MSI/Desktop/ARENA/data/data_test/conversation/pythonqa_test.jsonl"  # change this path

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def fuzzy(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

correct = 0
total = 0

print("Evaluating...")

with open(data_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        obj = json.loads(line)

        question = obj["question"]
        contexts = obj["contexts"]
        gold = obj["answer"]  # the correct context

        prompt = f"""Question: {question}

Contexts:
{contexts[0]}
{contexts[1]}

Réponse:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True
        )

        response = tokenizer.decode(out[0], skip_special_tokens=True)

        # extract only the generated part
        response = response[len(prompt):].strip()

        # Fuzzy matching with gold answer
        score = fuzzy(response.lower(), gold.lower())

        if score > 0.5:  # threshold (adjustable)
            correct += 1

        total += 1

print("=== RESULTS ===")
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", correct / total)
