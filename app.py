# =========================
# 1. ライブラリ
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# =========================
# 2. モデルロード（CPU用・軽量モデル）
# =========================
model_name = "rinna/japanese-gpt-1b"  # 認証不要の日本語モデル
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# =========================
# 3. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode="summary"):
    """
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    mode: "summary" / "bullet" / "check"
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

    # プロンプト作成
    prompt = f"{few_shot_examples}\n入力：{text}\n出力："

    # =========================
    # 4. モデル推論
    # =========================
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # =========================
    # 5. 出力検証・補正
    # =========================
    return parse_expense_output(output_text) if mode in ["summary", "bullet"] else check_expense_output(output_text)

# =========================
# 5a. 出力パース関数（安定版）
# =========================
def parse_expense_output(output_text):
    """
    モデル出力から「購入物」「理由」「申請区分」を抽出
    """
    # 最新の入力文のみ抽出
    last_input_match = re.search(r"入力：(.+?)\s*出力：", output_text, re.DOTALL)
    if last_input_match:
        relevant_text = output_text[last_input_match.end():]  # 出力部分だけ取得
    else:
        relevant_text = output_text

    result = {}
    for key in ["購入物", "理由", "申請区分"]:
        # 改行または文字列終端まででマッチ
        match = re.search(rf"{key}[:：]\s*(.*?)(?:\n|$)", relevant_text)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = "不明"
    return result

# =========================
# 5b. 簡易チェック関数
# =========================
def check_expense_output(output_text):
    """
    購入物が明確かどうかを「はい/いいえ」で返す
    """
    if "購入物" in output_text:
        return "はい"
    return "いいえ"

# =========================
# 6. 動作確認
# =========================
if __name__ == "__main__":
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

