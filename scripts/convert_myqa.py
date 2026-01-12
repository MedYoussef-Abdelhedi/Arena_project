import json, os, random, sys

def convert(source_json, out_train='data/data_direct/pythonqa/train.jsonl', out_val=None, out_test=None):
    with open(source_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    data = raw.get("questions") or raw  # adapte selon ton format
    random.seed(42)
    random.shuffle(data)
    total = len(data)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    train = data[:train_size]
    val = data[train_size:train_size+val_size]
    test = data[train_size+val_size:]

    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    with open(out_train, 'w', encoding='utf-8') as f:
        for item in train:
            # adapte champs question / reponse selon ton JSON
            question = item.get('question', '')
            answer = item.get('reponse_attendue', {}).get('raisonnement', '') + "\n" + item.get('reponse_attendue', {}).get('code', '')
            entry = {"prompt": question, "answers": [answer], "name":"pythonqa"}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("Wrote", out_train, len(train))

    def write_split(lst, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for item in lst:
                question = item.get('question','')
                answer = item.get('reponse_attendue', {}).get('raisonnement', '') + "\n" + item.get('reponse_attendue', {}).get('code', '')
                entry = {"question": question, "contexts": [f"Relevant: {answer}", "Irrelevant ..."], "answer": answer, "gold_indices": [0]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("Wrote", path, len(lst))

    if out_val:
        write_split(val, out_val)
    if out_test:
        write_split(test, out_test)

if __name__ == '__main__':
    src = sys.argv[1]
    convert(src, out_train='data/data_direct/pythonqa/train.jsonl', out_val='data/data_val/conversation/pythonqa_val.jsonl', out_test='data/data_test/conversation/pythonqa_test.jsonl')