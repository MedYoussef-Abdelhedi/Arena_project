import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

import torch

MODEL_PATH = "c:/Users/MSI/Desktop/ARENA/checkpoints/grpo_New"
TEST_FILE = "c:/Users/MSI/Desktop/ARENA/data/data_test/conversation/pythonqa_test.jsonl"

# === Sentence-BERT pour la similarité (tu peux changer le modèle) ===
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine(a, b):
    return float(util.pytorch_cos_sim(a, b)[0][0])

def evaluate():
    print("Loading Qwen...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True
    )

    results = []
    scores = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(x) for x in f]

    for item in tqdm(data):
        q = item["question"]
        gold = item["answer"]

        input_ids = tok(q, return_tensors="pt").to(model.device)

        output = model.generate(
            **input_ids,
            max_new_tokens=200,
            temperature=0.1
        )

        answer = tok.decode(output[0], skip_special_tokens=True)

        # similarité sémantique
        e1 = embedder.encode(answer, convert_to_tensor=True)
        e2 = embedder.encode(gold, convert_to_tensor=True)
        score = cosine(e1, e2)

        scores.append(score)

        results.append({
            "question": q,
            "gold_answer": gold,
            "model_response": answer,
            "similarity": score
        })

    print("\n=== FINAL SCORE ===")
    print("Mean similarity:", sum(scores)/len(scores))

    with open("eval_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    evaluate()
