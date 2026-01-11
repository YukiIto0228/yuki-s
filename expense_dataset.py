from datasets import Dataset
from transformers import AutoTokenizer

# ----------------------
# サンプル経費申請データ
# ----------------------
expense_ner_dataset = [
    {"text": "2026年1月10日、東京出張のため電車代として12000円、昼食代として1500円を申請します。",
     "labels": [
         (0, 11, "B-DATE"),
         (14, 16, "B-PURPOSE"),
         (18, 21, "B-PURPOSE"),
         (22, 27, "B-AMOUNT"),
         (28, 32, "B-PURPOSE"),
         (33, 38, "B-AMOUNT")
     ]},
    {"text": "2026年1月12日、大阪出張で新幹線代50000円、宿泊費10000円、会議費3000円を申請します。",
     "labels": [
         (0, 11, "B-DATE"),
         (13, 15, "B-PURPOSE"),
         (17, 20, "B-PURPOSE"),
         (20, 25, "B-AMOUNT"),
         (26, 30, "B-PURPOSE"),
         (30, 35, "B-AMOUNT"),
         (36, 39, "B-PURPOSE"),
         (39, 43, "B-AMOUNT")
     ]},
]

# ----------------------
# 使用するラベルセット
# ----------------------
label_list = ["O", "B-DATE", "B-PURPOSE", "B-AMOUNT"]
label_to_id = {l: i for i, l in enumerate(label_list)}

# ----------------------
# データセットをトークン化して TokenClassification 用に変換
# ----------------------
def char_labels_to_token_labels(example, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(example["text"], truncation=True, is_split_into_words=False, return_offsets_mapping=True)
    tokens = tokenized_inputs.tokens()
    labels = [label_to_id["O"]] * len(tokens)
    
    for start, end, label in example["labels"]:
        for i, (tok_start, tok_end) in enumerate(tokenized_inputs["offset_mapping"]):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            labels[i] = label_to_id[label]

    tokenized_inputs["labels"] = labels
    # offset_mappingは不要なので削除
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

def load_dataset(tokenizer):
    dataset = Dataset.from_list(expense_ner_dataset)
    dataset = dataset.map(lambda x: char_labels_to_token_labels(x, tokenizer, label_to_id))
    return dataset