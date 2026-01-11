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
    {"text": "2026年1月15日、名古屋出張の電車代として8000円、昼食代1000円を申請します。",
     "labels": [
         (0, 11, "B-DATE"),
         (14, 16, "B-PURPOSE"),
         (18, 21, "B-PURPOSE"),
         (22, 26, "B-AMOUNT"),
         (27, 31, "B-PURPOSE"),
         (32, 36, "B-AMOUNT")
     ]},
    {"text": "2026年1月18日、福岡出張で飛行機代40000円、宿泊費12000円、会議費5000円を申請します。",
     "labels": [
         (0, 11, "B-DATE"),
         (14, 16, "B-PURPOSE"),
         (18, 20, "B-PURPOSE"),
         (20, 25, "B-AMOUNT"),
         (26, 30, "B-PURPOSE"),
         (30, 35, "B-AMOUNT"),
         (36, 39, "B-PURPOSE"),
         (39, 44, "B-AMOUNT")
     ]},
    {"text": "2026年1月20日、札幌出張で電車代5000円、昼食代1200円を申請します。",
     "labels": [
         (0, 11, "B-DATE"),
         (14, 16, "B-PURPOSE"),
         (18, 21, "B-PURPOSE"),
         (22, 26, "B-AMOUNT"),
         (27, 31, "B-PURPOSE"),
         (32, 36, "B-AMOUNT")
     ]},
    {"text": "2026年1月22日、横浜出張で新幹線代30000円、宿泊費9000円を申請します。",
     "labels": [
         (0, 11, "B-DATE"),
         (14, 16, "B-PURPOSE"),
         (18, 21, "B-PURPOSE"),
         (22, 27, "B-AMOUNT"),
         (28, 32, "B-PURPOSE"),
         (33, 37, "B-AMOUNT")
     ]},
]


# ----------------------
# 使用するラベルセット
# ----------------------
label_list = ["O", "B-DATE", "B-PURPOSE", "B-AMOUNT"]
label_to_id = {l: i for i, l in enumerate(label_list)}

# 文字列ラベルを int に変換
def char_labels_to_token_labels(example, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        example["text"],
        truncation=True,
        is_split_into_words=False,
        return_offsets_mapping=True
    )
    
    labels = [label_to_id["O"]] * len(tokenized_inputs["input_ids"])
    
    for start, end, label_str in example["labels"]:
        for i, (tok_start, tok_end) in enumerate(tokenized_inputs["offset_mapping"]):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            labels[i] = label_to_id[label_str]  # 文字列 → int に変換

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

def load_dataset(tokenizer):
    dataset = Dataset.from_list(expense_ner_dataset)
    dataset = dataset.map(lambda x: char_labels_to_token_labels(x, tokenizer, label_to_id))

    return dataset

