# =========================
# 1. ライブラリ
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# =========================
# 2. モデルロード（軽量で誰でも使用可）
# =========================
model_name = "rinna/japanese-gpt-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # CPU/GPU自動割当
)

# =========================
# 3. 経費申請支援アプリ（ゼロショット形式）
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    """
    # =========================
    # プロンプト作成
    # =========================
    if mode in ["summary", "bullet"]:
        prompt = f"""以下の経費申請文から情報を抽出してください。
出力フォーマット:
購入物: ...
理由: ...
申請区分: ...

入力文:
{text}
出力:"""
    elif mode == "check":
        prompt = f"""以下の経費申請文の購入物が明確か確認してください。はい/いいえで答えてください。

入力文:
{text}
出力:"""
    else:
        return "エラー：mode が不正です"

    # =========================
    # モデル推論
    # =========================
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        repetition_penalty=1.2
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # =========================
    # 出力整形
    # =========================
    if mode in ["summary", "bullet"]:
        # 「購入物」「理由」「申請区分」を抽出
        result = {}
        for key in ["購入物", "理由", "申請区分"]:
            match = re.search(f"{key}[:：](.+)", output_text)
            result[key] = match.group(1).strip() if match else "不明"
        return result
    elif mode == "check":
        if "はい" in output_text:
            return "はい"
        elif "いいえ" in output_text:
            return "いいえ"
        else:
            return "不明"

# =========================
# 4. 動作確認
# =========================
sample_text = (
    "昨日、実験で急に必要になったため研究用ケーブルを購入しました。"
    "事前に申請を行う時間が取れなかったため、事後での申請となります。"
)

print("【要点抽出】")
print(expense_application_assistant(sample_text, "summary"))

print("\n【整理結果】")
print(expense_application_assistant(sample_text, "bullet"))

print("\n【確認結果】")
print(expense_application_assistant(sample_text, "check"))

