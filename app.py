from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# 2. モデルロード（LLM-JP日本語指示モデル）
# =========================
model_name = "llm-jp/llm-jp-3-8x1.8b-instruct3"  # 無料 & Transformersで直接呼び出し可

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # GPU使用時。CPUなら削除
)

# =========================
# 3. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文を対象に、情報抽出・確認支援を行う
    """

    if mode == "summary":
        prompt = (
            f"経費申請文: {text}\n"
            "抽出結果: 購入物:, 理由:, 申請区分:"
        )

    elif mode == "check":
        prompt = (
            f"経費申請文: {text}\n"
            "確認結果: 購入物は明確ですか:はい/いいえ, 理由は明確ですか:はい/いいえ, 申請区分は明確ですか:はい/いいえ"
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
# 4. 動作確認
# =========================
sample_text = (
    "昨日、実験で急に必要になったため研究用ケーブルを購入しました。"
    "事前に申請を行う時間が取れなかったため、事後での申請となります。"
)

print("【要点抽出】")
print(expense_application_assistant(sample_text, "summary"))

print("\n【確認結果】")
print(expense_application_assistant(sample_text, "check"))



