from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# =========================
# 1. モデルロード（TinyLLaMA 1.1B instruction-tuned）
# =========================
model_name = "jzhang38/tinyllama-1.1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",    # CPU環境でも自動配置
    torch_dtype=torch.float32,
    load_in_8bit=True     # CPU軽量化
)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

# =========================
# 2. 経費申請支援アプリ
# =========================
def expense_application_assistant(text, mode):
    """
    経費申請文を対象に、情報抽出・整理・確認支援を行う
    """

    prompts = {
        "summary": (
            "次の文章から経費申請に必要な情報を抜き出してください。\n"
            "購入物・理由・申請区分を箇条書きで出力してください。\n\n"
            f"文章:\n{text}\n\n要点:"
        ),
        "bullet": (
            "次の文章を経費申請用に整理してください。\n"
            "箇条書きで、購入物・購入理由・申請区分を出力してください。\n\n"
            f"文章:\n{text}\n\n整理結果:"
        ),
        "check": (
            "次の文章について情報の明確さを判断してください。\n"
            "購入物・理由・申請区分が明確か「はい/いいえ」で答えてください。\n\n"
            f"文章:\n{text}\n\n確認結果:"
        )
    }

    if mode not in prompts:
        return "エラー：mode が不正です"

    prompt = prompts[mode]

    output = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        repetition_penalty=1.2
    )

    # 出力整形
    return output[0]["generated_text"].strip()

# =========================
# 3. 動作確認
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



