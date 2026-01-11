# =========================
# 1. ライブラリ
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# =========================
# 2. モデルロード（CPU用軽量モデル）
# =========================
model_name = "rinna/japanese-gpt-1b"  # 認証不要・軽量モデル
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

# =========================
# 3. 経費申請支援アプリ（安定版）
# =========================
def expense_application_assistant(text, mode="summary"):
    """
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    mode: summary / bullet / check
    """

    # few-shot + 強制フォーマット
    few_shot = """
例1:
入力: 会議で使用するため文房具を購入しました。事前に申請済みです。
出力:
購入物: 文房具
理由: 会議で使用するため
申請区分: 事前申請

例2:
入力: 社外セミナー参加費として交通費と宿泊費を申請します。
出力:
購入物: 交通費、宿泊費
理由: 社外セミナー参加
申請区分: 事前申請
"""

    prompt = few_shot + f"\n入力: {text}\n出力:"

    # =========================
    # 4. モデル推論
    # =========================
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,   # 長すぎないように制限
        do_sample=False,     # 決定論的生成で安定
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # =========================
    # 5. 出力整形
    # =========================
    # few-shot部分を削除
    output_text = output_text.replace(few_shot, "").strip()

    # 改行で3行に分ける
    lines = [line.strip() for line in output_text.split("\n") if line.strip()]
    
    # 不足行を補完
    keys = ["購入物", "理由", "申請区分"]
    result = {}
    for i, key in enumerate(keys):
        if i < len(lines):
            # キーに沿った出力か簡易チェック
            if key not in lines[i]:
                result[key] = "不明"
            else:
                # 「購入物: 文房具」の形に整形
                result[key] = lines[i].split(":", 1)[1].strip()
        else:
            result[key] = "不明"

    return result

# =========================
# 6. 動作確認
# =========================
sample_text = (
    "昨日、実験で急に必要になったため研究用ケーブルを購入しました。"
    "事前に申請を行う時間が取れなかったため、事後での申請となります。"
)

print("【要点抽出】")
print(expense_application_assistant(sample_text, "summary"))


