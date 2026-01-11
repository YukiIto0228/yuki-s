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
# 3. バックオフィス支援アプリ
# =========================
def backoffice_text_organizer(text, mode):

    if mode == "summary":
        prompt = (
            "次の申請文を要点3つで簡潔に要約してください。\n"
            f"{text}"
        )

    elif mode == "bullet":
        prompt = (
            "次の文章をバックオフィス業務向けに箇条書きで整理してください。\n"
            f"{text}"
        )

    elif mode == "category":
        prompt = (
            "次の文章の業務カテゴリを1つ選び、理由を簡潔に述べてください。\n"
            "カテゴリ：経費申請、設備購入、問い合わせ、その他\n"
            f"{text}"
        )

    else:
        return "エラー：mode が不正です"

    output = generator(
        prompt,
        max_new_tokens=80
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
print(backoffice_text_organizer(sample_text, "summary"))

print("\n【箇条書き整理】")
print(backoffice_text_organizer(sample_text, "bullet"))

print("\n【カテゴリ推定】")
print(backoffice_text_organizer(sample_text, "category"))


