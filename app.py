# =========================
# 1. ライブラリ
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# =========================
# 2. モデルロード（CPUでも動く軽量モデル）
# =========================
model_name = "rinna/japanese-gpt-1b"  # Hugging Face認証不要の軽量日本語モデル
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # CPUでも動く
    torch_dtype=torch.float32
)

# =========================
# 3. モデル出力解析関数
# =========================
def parse_expense_output(output_text):
    """
    モデル出力から「購入物」「理由」「申請区分」を抽出
    """
    result = {}
    for key in ["購入物", "理由", "申請区分"]:
        # raw文字列 + 非貪欲マッチ
        match = re.search(rf"{key}[:：]\s*(.*?)(?=\s*(購入物|理由|申請区分|$))", output_text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = "不明"
    return result

# =========================
# 4. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode="summary"):
    """
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    mode:
      - summary : 要点抽出
      - bullet  : 箇条書き整理
      - check   : 確認（はい/いいえ）
    """
    # few-shot例
    few_shot_examples = """
例1:
入力：会議で使用するため文房具を購入しました。事前に申請済みです。
出力：
購入物: 文房具
理由: 会議で使用するため
申請区分: 事前申請

例2:
入力：社外セミナー参加費として交通費と宿泊費を申請します。
出力：
購入物: 交通費、宿泊費
理由: 社外セミナー参加
申請区分: 事前申請

例3:
入力：実験用ケーブルを購入しました。急ぎのため事後申請です。
出力：
購入物: ケーブル
理由: 実験用
申請区分: 事後申請
"""

    # プロンプト作成
    if mode in ["summary", "bullet"]:
        prompt = f"{few_shot_examples}\n入力：{text}\n出力："
    elif mode == "check":
        prompt = f"{few_shot_examples}\n入力：{text}\n出力：購入物は明確ですか："
    else:
        return "エラー：mode が不正です"

    # =========================
    # 5. モデル推論
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
    # 6. 出力整理
    # =========================
    if mode in ["summary", "bullet"]:
        return parse_expense_output(output_text)
    elif mode == "check":
        # 「はい/いいえ」で応答できるよう補完
        if "はい" in output_text:
            return "はい"
        elif "いいえ" in output_text:
            return "いいえ"
        else:
            return "不明"

# =========================
# 7. 動作確認
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


