# expense_checker.py
import re
import torch
from transformers import BertTokenizer, BertForTokenClassification
import streamlit as st

# ----------------------
# STEP1: 経費情報抽出
# ----------------------

# サンプルラベル
LABELS = ["O", "B-AMOUNT", "B-PURPOSE", "B-DATE"]

# モデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
model = BertForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese-v3", num_labels=len(LABELS))

def extract_entities(text):
    """簡易的なNER推論（デモ用）"""
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
# STEP2: 合計金額計算
# ----------------------
def sum_amount(entities):
    """抽出された金額の合計"""
    total = 0
    for token, label in entities:
        if label == "B-AMOUNT":
            # 数字だけ抽出
            nums = re.findall(r"\d+", token)
            if nums:
                total += int(nums[0])
    return total

# ----------------------
# STEP3: 規程チェック
# ----------------------
MAX_AMOUNT = 50000  # 例：上限5万円

def check_rules(entities):
    reasons = []
    total = sum_amount(entities)
    if total > MAX_AMOUNT:
        reasons.append(f"合計金額({total}円)が上限({MAX_AMOUNT}円)を超えています")
    # 用途チェック例
    for token, label in entities:
        if label == "B-PURPOSE" and "遊び" in token:
            reasons.append("遊興費は不可")
    return reasons

# ----------------------
# Streamlit UI
# ----------------------
st.title("経費申請妥当性チェック (デモ版)")

text = st.text_area("申請内容を入力してください:")

if st.button("チェック実行"):
    entities = extract_entities(text)
    st.write("抽出結果:", entities)

    total = sum_amount(entities)
    st.write(f"合計金額: {total}円")

    reasons = check_rules(entities)
    if reasons:
        st.write("妥当性チェック: NG")
        for r in reasons:
            st.write("-", r)
    else:
        st.write("妥当性チェック: OK")
