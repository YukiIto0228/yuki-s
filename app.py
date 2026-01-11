from transformers import pipeline


# =========================
# 2. モデルロード（T5日本語）
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
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    """

    if mode == "summary":
        prompt = (
            "次の例にならって、経費申請文から情報を抜き出してください。\n\n"
            "【例】\n"
            "入力：会議で使用するため文房具を購入しました。事前に申請済みです。\n"
            "出力：\n"
            "購入物：文房具\n"
            "理由：会議で使用するため\n"
            "申請区分：事前申請\n\n"
            "【入力】\n"
            f"{text}\n\n"
            "【出力】\n"
        )

    elif mode == "bullet":
        prompt = (
            "次の例にならって、経費申請文を整理してください。\n\n"
            "【例】\n"
            "入力：会議で使用するため文房具を購入しました。事前に申請済みです。\n"
            "出力：\n"
            "・購入物：文房具\n"
            "・購入理由：会議で使用するため\n"
            "・申請区分：事前申請\n\n"
            "【入力】\n"
            f"{text}\n\n"
            "【出力】\n"
        )

    elif mode == "check":
        prompt = (
            "次の例にならって判断してください。\n\n"
            "【例】\n"
            "入力：会議で使用するため文房具を購入しました。事前に申請済みです。\n"
            "出力：\n"
            "購入物は明確ですか：はい\n"
            "理由は明確ですか：はい\n"
            "申請区分は明確ですか：はい\n\n"
            "【入力】\n"
            f"{text}\n\n"
            "【出力】\n"
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

print("【要点抽出】")
print(expense_application_assistant(sample_text, "summary"))

print("\n【整理結果】")
print(expense_application_assistant(sample_text, "bullet"))

print("\n【確認結果】")
print(expense_application_assistant(sample_text, "check"))


