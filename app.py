# =========================
# 1. ライブラリ
# =========================
from transformers import pipeline


# =========================
# 2. モデルロード（T5系）
# =========================
generator = pipeline(
    "text2text-generation",
    model="sonoisa/t5-base-japanese"
)


# =========================
# 3. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文から必要情報を抽出・整理する関数
    """

    if mode == "summary":
        # 情報抽出型プロンプト
        prompt = (
            "次の経費申請文から情報を抜き出してください。\n"
            "購入物：\n"
            "理由：\n"
            "申請区分（事前／事後）：\n"
            f"{text}"
        )

    elif mode == "bullet":
        # 定型フォーマットへの書き換え
        prompt = (
            "次の経費申請文を、経理担当者向けに整理して書き換えてください。\n"
            "・購入物：\n"
            "・購入理由：\n"
            "・申請区分（事前／事後）：\n"
            f"{text}"
        )

    elif mode == "check":
        # 判断型（YES / NO）
        prompt = (
            "次の経費申請文について判断してください。\n"
            "購入物は明確ですか：はい／いいえ\n"
            "理由は明確ですか：はい／いいえ\n"
            "事前申請か事後申請か明確ですか：はい／いいえ\n"
            f"{text}"
        )

    else:
        return "エラー：mode が不正です"

    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.2
    )

    return output[0]["generated_text"]


# =========================
# 4. 動作確認
# =========================
sample_text = (
    "昨日、実験で急に必要になったため研究用ケーブルを購入しました。"
    "事前に申請を行う時間が取れなかったため、事後での申請となります。"
)

print("【要点要約】")
print(expense_application_assistant(sample_text, "summary"))

print("\n【箇条書き整理】")
print(expense_application_assistant(sample_text, "bullet"))

print("\n【確認事項】")
print(expense_application_assistant(sample_text, "check"))


