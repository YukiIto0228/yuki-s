from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="sonoisa/t5-base-japanese"
)

# 1. 内部ウォームアップ用の例文（表示はしない）
dummy_examples = [
    "会議で使用するため文房具を購入しました。事前に申請済みです。",
    "急遽必要になった実験用ケーブルを購入しました。事後申請です。"
]

# 例文を内部的に生成させてモデルにタスク形式を認識させる
for example in dummy_examples:
    _ = generator(
        f"経費申請文から購入物・理由・申請区分を抽出してください：{example}",
        max_new_tokens=10,  # 出力はどうでもよいので小さく
        do_sample=False
    )

# 2. CLI入力や実際のタスク入力
def expense_task(text, mode):
    if mode == "summary":
        prompt = f"経費申請文から購入物・理由・申請区分を抽出してください：{text}"
    elif mode == "check":
        prompt = f"経費申請文の情報が明確か判定してください：{text}"
    else:
        return "mode error"

    output = generator(prompt, max_new_tokens=80, do_sample=False)
    return output[0]["generated_text"].strip()

# 実行例
user_input = "昨日、急に必要になったため研究用ケーブルを購入しました。事後申請です。"

print("【要点抽出】")
print(expense_task(user_input, "summary"))

print("\n【確認結果】")
print(expense_task(user_input, "check"))



