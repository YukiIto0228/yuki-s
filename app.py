
from transformers import pipeline

# =========================
# モデルロード（T5日本語）
# =========================
generator = pipeline(
    "text2text-generation",
    model="sonoisa/t5-base-japanese"
)

# =========================
# 内部ウォームアップ（例文でタスク形式を理解させる）
# =========================
dummy_examples = [
    "会議で使用するため文房具を購入しました。事前に申請済みです。",
    "急遽必要になった実験用ケーブルを購入しました。事後申請です。"
]

for example in dummy_examples:
    # 出力は破棄、モデルにタスク形式を暗黙的に覚えさせるだけ
    _ = generator(
        f"経費申請文から購入物・理由・申請区分を抽出してください。"
        f"出力形式は '購入物:XXX, 理由:XXX, 申請区分:XXX' にしてください。\n"
        f"{example}",
        max_new_tokens=5,
        do_sample=False
    )

# =========================
# 経費申請支援アプリ（CLI用）
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文を対象に要点抽出・確認を行う
    """

    if mode == "summary":
        prompt = (
            f"次の経費申請文から購入物と理由、申請区分を抽出してください。"
            f"出力形式は '購入物:XXX, 理由:XXX, 申請区分:XXX' にしてください。\n"
            f"経費申請文: {text}\n出力:"
        )

    elif mode == "check":
        prompt = (
            f"次の経費申請文の情報が明確かどうか判定してください。"
            f"出力形式は '購入物は明確ですか:はい/いいえ, 理由は明確ですか:はい/いいえ, 申請区分は明確ですか:はい/いいえ' にしてください。\n"
            f"経費申請文: {text}\n出力:"
        )

    else:
        return "エラー：mode が不正です"

    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.2
    )

    return output[0]["generated_text"].strip()

# =========================
# CLI入力例
# =========================
if __name__ == "__main__":
    user_input = input("経費申請文を入力してください:\n")

    print("\n【要点抽出】")
    print(expense_application_assistant(user_input, "summary"))

    print("\n【確認結果】")
    print(expense_application_assistant(user_input, "check"))


