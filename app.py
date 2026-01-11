import re
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="sonoisa/t5-base-japanese"
)

def expense_application_assistant(text, mode):
    if mode == "summary":
        prompt = (
            f"経費申請文から購入物、理由、申請区分を抽出してください。\n{text}\n"
        )
    elif mode == "bullet":
        prompt = (
            f"経費申請文を整理して箇条書きにしてください。\n{text}\n"
        )
    elif mode == "check":
        prompt = (
            f"経費申請文の内容が明確かを判断してください。購入物・理由・申請区分ごとに「はい/いいえ」で答えてください。\n{text}\n"
        )
    else:
        return "エラー：mode が不正です"

    output = generator(prompt, max_new_tokens=80, do_sample=False, repetition_penalty=1.2)
    result = output[0]["generated_text"]

    # 不要なプレフィックスや余計な文字列を削除
    result = re.sub(r"【入力】|【出力】", "", result)
    result = result.strip()

    return result

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


