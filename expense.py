# expense_cli.py
import re
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification

# ----------------------
# STEP0: ラベル定義
# ----------------------
LABELS = ["O", "B-AMOUNT", "B-PURPOSE", "B-DATE"]

# ----------------------
# STEP1: モデル準備
# ----------------------
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
model = BertForTokenClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-v3",
    num_labels=len(LABELS)
)
model.eval()  # 推論モード

# ----------------------
# STEP2: 経費情報抽出
# ----------------------
def extract_entities(text):
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    entities = []
    for token, label_id in zip(tokens, predictions):
        label = LABELS[label_id]
        if label != "O":
            entities.append((token, label))
    return entities

# ----------------------
# STEP3: 合計金額計算
# ----------------------
def sum_amount(entities):
    total = 0
    for token, label in entities:
        if label == "B-AMOUNT":
            nums = re.findall(r"\d+", token)
            if nums:
                total += int(nums[0])
    return total

# ----------------------
# STEP4: 規程チェック
# ----------------------
MAX_AMOUNT = 50000  # 上限例

def check_rules(entities):
    reasons = []
    total = sum_amount(entities)
    if total > MAX_AMOUNT:
        reasons.append(f"合計金額({total}円)が上限({MAX_AMOUNT}円)を超えています")
    # 簡易用途チェック
    for token, label in entities:
        if label == "B-PURPOSE" and "遊び" in token:
            reasons.append("遊興費は不可")
    return reasons

# ----------------------
# STEP5: CLI入力
# ----------------------
if __name__ == "__main__":
    print("=== 経費申請妥当性チェック CLI版 ===")
    text = input("申請内容を入力してください:\n> ")

    entities = extract_entities(text)
    print("\n--- 抽出結果 ---")
    for token, label in entities:
        print(f"{token}: {label}")

    total = sum_amount(entities)
    print(f"\n合計金額: {total}円")

    reasons = check_rules(entities)
    print("\n--- 妥当性チェック ---")
    if reasons:
        print("NG (確認が必要)")
        for r in reasons:
            print("-", r)
    else:
        print("OK (問題なし)")


