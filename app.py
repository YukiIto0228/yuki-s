from transformers import pipeline
import re

# =========================
# モデルロード（T5日本語・抽出特化）
# =========================
generator = pipeline(
    "text2text-generation",
    model="retrieva-jp/t5-base-long"
)

# =========================
# 抽出処理
# =========================
def extract_expense_info(text: str) -> str:
    """
    経費申請文から
    ・購入物
    ・理由
    ・申請区分
    を抽出する
    """

    prompt = (
        "task: 経費申請情報抽出\n"
        "format:\n"
        "購入物=<内容>\n"
        "理由=<内容>\n"
        "申請区分=<事前|事後>\n"
        f"text: {text}\n"
        "output:\n"
    )

    output = generator(
        prompt,
        max_new_tokens=64,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    return output[0]["generated_text"].strip()


# =========================
# パース処理（LLMを信用しない）
# =========================
def parse_extracted(text: str) -> dict:
    """
    抽出結果を安全にパース
    """
    result = {
        "購入物": "",
        "理由": "",
        "申請区分": ""
    }

    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in result:
                result[key] = value

    return result


# =========================
# 明確性チェック（ルールベース）
# =========================
def check_clarity(parsed: dict) -> dict:
    """
    明確性はLLMに聞かず、ルールで判定
    """
    def is_clear(value: str) -> str:
        return "はい" if value and len(value) >= 2 else "いいえ"

    return {
        "購入物は明確ですか": is_clear(parsed["購入物"]),
        "理由は明確ですか": is_clear(parsed["理由"]),
        "申請区分は明確ですか": (
            "はい" if parsed["申請区分"] in ["事前", "事後"] else "いいえ"
        )
    }


# =========================
# 経費申請支援アプリ（CLI）
# =========================
def expense_application_assistant(text: str):
    extracted_text = extract_expense_info(text)
    parsed = parse_extracted(extracted_text)
    clarity = check_clarity(parsed)

    return extracted_text, clarity


# =========================
# CLI実行
# =========================
if __name__ == "__main__":
    user_input = input("経費申請文を入力してください:\n")

    extracted, check_result = expense_application_assistant(user_input)

    print("\n【要点抽出結果】")
    print(extracted)

    print("\n【明確性チェック】")
    for k, v in check_result.items():
        print(f"{k}: {v}")


