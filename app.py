from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="rinna/japanese-gpt2-medium"
)


# =========================
# 3. バックオフィス支援アプリ本体
# =========================
def backoffice_text_organizer(text, mode):
    """
    バックオフィス自由記述文を整理するアプリ関数

    Parameters
    ----------
    text : str
        申請文・問い合わせ文などの自由記述
    mode : str
        'summary'  : 要点を3点で要約
        'bullet'   : 業務向け箇条書き整理
        'category' : 業務カテゴリ推定

    Returns
    -------
    str
        整形後の文章
    """

    if mode == "summary":
        prompt = (
            "以下の申請文の要点を3点で簡潔にまとめてください。\n"
            f"申請文：{text}"
        )

    elif mode == "bullet":
        prompt = (
            "以下の文章を、バックオフィス業務で読みやすいように"
            "箇条書きで整理してください。\n"
            f"文章：{text}"
        )

    elif mode == "category":
        prompt = (
            "次の文章が該当する業務カテゴリを1つ選び、理由も述べてください。\n"
            "カテゴリ例：経費申請、設備購入、問い合わせ、その他\n"
            f"文章：{text}"
        )

    else:
        return "エラー：mode が不正です"

    # 推論のみ（ファインチューニングなし）
    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7
    )

    return output[0]["generated_text"]


# =========================
# 4. アプリの使用例（デモ）
# =========================
sample_text = (
    "昨日、実験で急に必要になったため研究用ケーブルを購入しました。"
    "事前に申請を行う時間が取れなかったため、事後での申請となります。"
)

print("【要点要約】")
print(backoffice_text_organizer(sample_text, "summary"))

print("\n【箇条書き整理】")
print(backoffice_text_organizer(sample_text, "bullet"))

print("\n【カテゴリ推定】")
print(backoffice_text_organizer(sample_text, "category"))
