# eval_to_arena_format.py
import argparse
import json
import os
import re
import difflib
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- helpers --------------------
def normalize_text(t):
    if t is None:
        return ""
    return " ".join(str(t).lower().strip().split())

def fuzzy_score(a, b):
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if b_n in a_n or a_n in b_n:
        return 1.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()

def extract_tags(text):
    """Extract <relevance>, <analysis>, <answer> from text (robustly)."""
    if text is None:
        return "", "", ""
    s = text
    def get_tag(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        # fallback: try single-line pattern without explicit closing
        m2 = re.search(rf"<{tag}>(.*)", s, flags=re.IGNORECASE | re.DOTALL)
        return m2.group(1).strip() if m2 else ""
    return get_tag("relevance"), get_tag("analysis"), get_tag("answer")

# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="c:/Users/MSI/Desktop/ARENA/checkpoints/grpo_New")
    p.add_argument("--data", required=True, help="c:/Users/MSI/Desktop/ARENA/data/data_test/conversation/pythonqa_test.jsonl")
    p.add_argument("--output", default="arena_results.jsonl", help="Output JSONL file")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--fuzzy-threshold", type=float, default=0.65)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading tokenizer and model from:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    # read dataset
    items = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if args.max_samples:
        items = items[: args.max_samples]
    print("Samples to evaluate:", len(items))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "w", encoding="utf-8")

    total = 0
    matched = 0
    sims = []

    for idx, item in enumerate(tqdm(items, desc="Eval")):
        total += 1
        q = item.get("question", "")
        contexts = item.get("contexts", []) or []
        gold = item.get("answer", "")  # expected "gold" (the arena expected output)
        # build context string: include contexts enumerated
        ctx_str = ""
        for i, c in enumerate(contexts):
            ctx_str += f"[Context {i}]:\n{c}\n\n"

        # Strict prompt template - force tag output
        prompt = (
            f"Tu es un assistant. Pour l'exemple ci-dessous, RÉPONDS STRICTEMENT au format suivant (ne rien ajouter) :\n\n"
            f"<relevance>INDEX</relevance>\n"
            f"<analysis>TEXTE_BRÈVE_JUSTIFIANT_LE_CHOIX</analysis>\n"
            f"<answer>RÉPONSE_FINALE</answer>\n\n"
            f"Consignes:\n"
            f"- <relevance> doit contenir l'indice (0-based) du contexte le plus pertinent parmi les Contexts fournis, ou -1 si aucun.\n"
            f"- <analysis> : une justification courte (1-2 phrases).\n"
            f"- <answer> : la réponse finale, sans code block ni balises. Écris uniquement la réponse.\n"
            f"- Ne mets rien d'autre hors des 3 balises.\n\n"
            f"Question:\n{q}\n\n"
            f"{ctx_str}\n"
            f"Réponds maintenant en respectant STRICTEMENT le format ci-dessus.\n"
        )

        # tokenize & generate (safe extraction by slicing token ids)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # ensure pad token exists
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            # deterministic generation (do_sample=False). If you want sampling: set do_sample=True
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                # temperature ignored if do_sample=False
            )

        # extract only newly generated ids (avoid keeping the prompt)
        gen_ids = out[0][input_len:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Parse tags
        relevance_s, analysis_s, answer_s = extract_tags(generated)

        # compute similarity between generated answer and gold
        sim = fuzzy_score(answer_s, gold)
        sims.append(sim)
        is_match = sim >= args.fuzzy_threshold
        if is_match:
            matched += 1

        # write result
        result = {
            "id": item.get("id", idx),
            "question": q,
            "contexts": contexts,
            "gold_answer": gold,
            "generated_raw": generated,
            "generated_relevance": relevance_s,
            "generated_analysis": analysis_s,
            "generated_answer": answer_s,
            "similarity": sim,
            "match": bool(is_match),
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    out_f.close()

    mean_sim = sum(sims) / len(sims) if sims else 0.0
    acc = matched / total if total else 0.0
    print("\n===== SUMMARY =====")
    print("Samples:", total)
    print("Mean similarity:", mean_sim)
    print(f"Matches (sim>={args.fuzzy_threshold}): {matched} ({acc:.3%})")
    print("Results saved to:", args.output)
    print("===================")

if __name__ == "__main__":
    main()
