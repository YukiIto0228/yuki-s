from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# ----------------------
# 1. ラベルセット
# ----------------------
label_list = ["O", "B-DATE", "B-PURPOSE", "B-AMOUNT"]
label_to_id = {l: i for i, l in enumerate(label_list)}

# ----------------------
# 2. サンプル経費データ（6 件）
# ----------------------
expense_ner_dataset = [
    {"text": "2026年1月10日、東京出張のため電車代として12000円、昼食代として1500円を申請します。",
     "labels": [(0, 11, "B-DATE"), (14, 16, "B-PURPOSE"), (18, 21, "B-PURPOSE"),
                (22, 27, "B-AMOUNT"), (28, 32, "B-PURPOSE"), (33, 38, "B-AMOUNT")]},
    {"text": "2026年1月12日、大阪出張で新幹線代50000円、宿泊費10000円、会議費3000円を申請します。",
     "labels": [(0, 11, "B-DATE"), (13, 15, "B-PURPOSE"), (17, 20, "B-PURPOSE"),
                (20, 25, "B-AMOUNT"), (26, 30, "B-PURPOSE"), (30, 35, "B-AMOUNT"),
                (36, 39, "B-PURPOSE"), (39, 43, "B-AMOUNT")]},
    {"text": "2026年1月15日、名古屋出張の電車代として8000円、昼食代1000円を申請します。",
     "labels": [(0, 11, "B-DATE"), (14, 16, "B-PURPOSE"), (18, 21, "B-PURPOSE"),
                (22, 26, "B-AMOUNT"), (27, 31, "B-PURPOSE"), (32, 36, "B-AMOUNT")]},
    {"text": "2026年1月18日、福岡出張で飛行機代40000円、宿泊費12000円、会議費5000円を申請します。",
     "labels": [(0, 11, "B-DATE"), (14, 16, "B-PURPOSE"), (18, 20, "B-PURPOSE"),
                (20, 25, "B-AMOUNT"), (26, 30, "B-PURPOSE"), (30, 35, "B-AMOUNT"),
                (36, 39, "B-PURPOSE"), (39, 44, "B-AMOUNT")]},
    {"text": "2026年1月20日、札幌出張で電車代5000円、昼食代1200円を申請します。",
     "labels": [(0, 11, "B-DATE"), (14, 16, "B-PURPOSE"), (18, 21, "B-PURPOSE"),
                (22, 26, "B-AMOUNT"), (27, 31, "B-PURPOSE"), (32, 36, "B-AMOUNT")]},
    {"text": "2026年1月22日、横浜出張で新幹線代30000円、宿泊費9000円を申請します。",
     "labels": [(0, 11, "B-DATE"), (14, 16, "B-PURPOSE"), (18, 21, "B-PURPOSE"),
                (22, 27, "B-AMOUNT"), (28, 32, "B-PURPOSE"), (33, 37, "B-AMOUNT")]},
]

# ----------------------
# 3. ラベルを int に変換
# ----------------------
expense_ner_dataset_int = []
for ex in expense_ner_dataset:
    new_labels = [(start, end, label_to_id[label_str]) for start, end, label_str in ex["labels"]]
    expense_ner_dataset_int.append({"text": ex["text"], "labels": new_labels})

# ----------------------
# 4. Dataset 作成
# ----------------------
dataset = Dataset.from_list(expense_ner_dataset_int)

# ----------------------
# 5. Tokenizer
# ----------------------
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def char_labels_to_token_labels(example, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        example["text"],
        truncation=True,
        is_split_into_words=False,
        return_offsets_mapping=True
    )
    
    labels = [label_to_id["O"]] * len(tokenized_inputs["input_ids"])
    
    for start, end, label_id in example["labels"]:
        for i, (tok_start, tok_end) in enumerate(tokenized_inputs["offset_mapping"]):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            labels[i] = label_id

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

dataset = dataset.map(lambda x: char_labels_to_token_labels(x, tokenizer, label_to_id))

# ----------------------
# 6. train/test split
# ----------------------
dataset = dataset.train_test_split(test_size=0.33)  # 6 件 → train 4 / test 2
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ----------------------
# 7. モデル
# ----------------------
num_labels = len(label_list)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# ----------------------
# 8. Trainer 準備
# ----------------------
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir="./ner_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# ----------------------
# 9. 学習開始
# ----------------------
trainer.train()
trainer.save_model("./ner_finetuned")
print("Fine-tuned model saved to ./ner_finetuned")

