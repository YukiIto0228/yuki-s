# =========================
# 1. ライブラリ
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# =========================
# 2. モデルロード（CPU, 量子化なし）
# =========================
model_name = "rinna/japanese-gpt-1b"  # 認証不要・誰でも使える軽量モデル
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"  # CPUで動かす
)

# =========================
# 3. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    """
    few_shot_examples = """
例1:
入力：会議で使用するため文房具を購入しました。事前に申請済みです。
出力：
購入物：文房具
理由：会議で使用するため
申請区分：事前申請

例2:
入力：社外セミナー参加費として交通費と宿泊費を申請します。
出力：
購入物：交通費、宿泊費
理由：社外セミナー参加
申請区分：事前申請

例3:
入力：実験用ケーブルを購入しました。急ぎのため事後申請です。
出力：
購入物：ケーブル
理由：実験用
申請区分：事後申請
"""

    if mode == "summary":
        prompt = f"{few_shot_examples}\n入力：{text}\n出力："
    elif mode == "bullet":
        prompt = f"{few_shot_examples}\n入力：{text}\n出力："
    elif mode == "check":
        prompt = f"{few_shot_examples}\n入力：{text}\n出力：購入物は明確ですか："
    else:
        return "エラー：mode が不正です"

    # =========================
    # 4. モデル推論
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
    # 5. 出力検証・補正
    # =========================
    if mode in ["summary", "bullet"]:
        if not re.search(r"購入物", output_text):
            output_text += "\n購入物: 不明"
        if not re.search(r"申請区分", output_text):
            output_text += "\n申請区分: 不明"

    elif mode == "check":
        if "はい" not in output_text and "いいえ" not in output_text:
            output_text += " 不明"

    return output_text

# =========================
# 6. 動作確認
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



