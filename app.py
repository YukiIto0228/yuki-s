from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="sonoisa/t5-base-japanese"
)


# =========================
# 3. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文を対象に、要約・整理・判断支援を行う関数
    """

    if mode == "summary":
        # 経理担当者が即判断できる要点要約
        prompt = (
            "次の経費申請文について、"
            "経理担当者が確認すべき要点を3点で簡潔に要約してください。\n"
            "（何を購入したか、理由、事前申請か事後申請か）\n"
            f"{text}"
        )

    elif mode == "bullet":
        # 記載揺れを抑えるための構造化
        prompt = (
            "次の経費申請文を、経理担当者向けに箇条書きで整理してください。\n"
            "・購入物\n"
            "・購入理由\n"
            "・申請タイミング（事前／事後）\n"
            f"{text}"
        )

    elif mode == "check":
        # 規程確認・差し戻し判断の補助
        prompt = (
            "次の経費申請文について、"
            "確認が必要な点や不明点があれば簡潔に指摘してください。\n"
            f"{text}"
        )

    else:
        return "エラー：mode が不正です"

    output = generator(
        prompt,
        max_new_tokens=100
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



