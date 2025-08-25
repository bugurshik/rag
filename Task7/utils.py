import json

def save_as_jsonl(file_target:str, logs):
    with open(file_target, 'w', encoding='utf-8') as f:
        for log_entry in logs:
            json_line = json.dumps(log_entry, ensure_ascii=False)
            f.write(json_line + '\n')

def read_jsonl(file_target:str):
    data = []
    with open(file_target, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def read_as_list(file_target:str):
    with open(file_target, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]
