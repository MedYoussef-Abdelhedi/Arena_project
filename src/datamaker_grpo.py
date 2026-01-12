import json
import argparse
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', nargs='+', default=['hotpotqa'], help='List of dataset names (e.g., hotpotqa two_wiki musique)')
    parser.add_argument('--saved-name', type=str, default='grpo_25000', help='Output file name (without extension)')
    parser.add_argument('--trainfile-name', type=str, default='train', help='Input training data file name (without extension)')
    args = parser.parse_args()

    dataset_names = args.name
    saved_name = args.saved_name
    trainfile_name = args.trainfile_name

    all_data = []

    # Load and merge data from specified dataset names
    for name in dataset_names:
        data_path = f'C:/Users/MSI/Desktop/ARENA/data/data_direct/{name}/{trainfile_name}.jsonl'
        print(f'Loading data from: {data_path}')
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))

    # Shuffle the merged dataset
    random.seed(123456)
    random.shuffle(all_data)

    dialogs = []
    for i, item in enumerate(all_data):
        dialog = {
            "prompt": [{"role": "user", "content": item['prompt']}],
            "answers": item['answers'],
            "supporting_ids": item.get('supporting_ids', []),
            "name": item['name'],
            "id": i,
        }
        dialogs.append(dialog)

    print(f'Number of dialogs: {len(dialogs)}')

    # Ensure output directory exists
    output_dir = 'C:/Users/MSI/Desktop/ARENA/data/data_train/grpo'
    os.makedirs(output_dir, exist_ok=True)

    # Save the formatted dialog data
    output_path = os.path.join(output_dir, f'{saved_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for dialog in dialogs:
            f.write(json.dumps(dialog, ensure_ascii=False) + '\n')
