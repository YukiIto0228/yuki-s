from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from expense_dataset import load_dataset, label_list

# ----------------------
# Tokenizer & Dataset
# ----------------------
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(tokenizer)
# train/test分割
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ----------------------
# モデル
# ----------------------
num_labels = len(label_list)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# ----------------------
# Trainer 準備
# ----------------------
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir="./ner_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs"
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
# 学習開始
# ----------------------
trainer.train()
trainer.save_model("./ner_finetuned")
print("Fine-tuned model saved to ./ner_finetuned")
