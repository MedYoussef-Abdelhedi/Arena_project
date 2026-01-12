# scripts/eval_infer.py
import json, torch, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT = "checkpoints/grpo/checkpoint-10"  # adapte si besoin
TEST_FILE = "data/data_test/conversation/pythonqa_test.jsonl"  # ou ton test
OUT_FILE = "results/grpo_eval_preds.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device)
model.eval()

def generate_text(prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

items = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    #!/usr/bin/env python
    # scripts/eval_infer.py
    import argparse
    import json
    import os
    import sys
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Evaluating GRPO model...")

    parser = argparse.ArgumentParser(description="Evaluate a local checkpoint for GRPO model")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/grpo/checkpoint-5", help="Path to local checkpoint folder")
    parser.add_argument("--test-file", "-t", default="data/data_test/conversation/pythonqa_test.jsonl", help="Test jsonl file")
    parser.add_argument("--out-file", "-o", default="results/grpo_eval_preds.json", help="Output predictions file")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    TEST_FILE = args.test_file
    OUT_FILE = args.out_file
    MAX_NEW_TOKENS = args.max_new_tokens

    # Validate paths
    if not os.path.isdir(CHECKPOINT):
        print(f"ERROR: checkpoint folder not found: {CHECKPOINT}")
        print("Available local checkpoints under 'checkpoints/grpo/':")
        try:
            for name in sorted(os.listdir("checkpoints/grpo")):
                print(" -", name)
        except Exception:
            pass
        sys.exit(1)

    if not os.path.isfile(TEST_FILE):
        print(f"ERROR: test file not found: {TEST_FILE}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    except Exception as e:
        print("Failed to load tokenizer from checkpoint:", CHECKPOINT)
        print(e)
        sys.exit(1)

    try:
        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print("Failed to load model from checkpoint:", CHECKPOINT)
        print(e)
        sys.exit(1)


    def generate_text(prompt, max_new_tokens=64):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # Read test items
    items = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line[:200])

    if not items:
        print("No test items found. Exiting.")
        sys.exit(0)

    # Generate
    results = []
    for it in tqdm(items, desc="Generating"):
        q = it.get("question") or it.get("prompt") or it.get("input") or ""
        gold = it.get("answer") or it.get("output") or it.get("response", "")
        pred = generate_text(q, max_new_tokens=MAX_NEW_TOKENS)
        results.append({"question": q, "gold": gold, "pred": pred})

    # save results
    os.makedirs(os.path.dirname(OUT_FILE) or "results", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fo:
        json.dump(results, fo, indent=2, ensure_ascii=False)
    print("Saved:", OUT_FILE)

    # Print samples
    if results:
        print("Sample predictions:")
        for r in results[:3]:
            print("Q:", r["question"])
            print("P:", r["pred"])
            print("G:", r["gold"])