# scripts/compute_metrics.py
import json, re
def normalize(s):
    if s is None: return ""
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def f1_score(gold, pred):
    g = normalize(gold).split()
    p = normalize(pred).split()
    if not g or not p: return 0.0
    common = set(g) & set(p)
    if len(common)==0: return 0.0
    precision = len(common)/len(p)
    recall = len(common)/len(g)
    if precision + recall == 0: return 0.0
    return 2*precision*recall/(precision+recall)

res = json.load(open("results/grpo_eval_preds.json","r",encoding="utf-8"))
ems = sum(1 for r in res if normalize(r["gold"]) == normalize(r["pred"]))
avg_f1 = sum(f1_score(r["gold"], r["pred"]) for r in res)/len(res)
print(f"Exact Match: {ems}/{len(res)} ({ems/len(res):.4f})")
print(f"Avg F1: {avg_f1:.4f}")
# save a csv for manual inspection
import csv
with open("results/grpo_eval_table.csv","w",newline="",encoding="utf-8") as cf:
    w = csv.writer(cf)
    w.writerow(["question","gold","pred","match"])
    for r in res:
        w.writerow([r["question"], r["gold"], r["pred"], normalize(r["gold"])==normalize(r["pred"])])
print("Wrote results/grpo_eval_table.csv")