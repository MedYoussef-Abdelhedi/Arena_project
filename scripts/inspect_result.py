# Python quick inspector: shows first 5 lines from results file
import json
f="c:/Users/MSI/Desktop/ARENA/results/inference_results.jsonl"
for i,line in enumerate(open(f,encoding='utf-8')):
    if i>=5: break
    obj=json.loads(line)
    print("=== ITEM",i,"===")
    print("gold_answer:",repr(obj.get("gold_answer")))
    print("model_response:",repr(obj.get("model_response")[:400]))
    print()

