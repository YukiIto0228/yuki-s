import re
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "rinna/japanese-gpt-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def expense_application_assistant(text):
    few_shot = """
例1:
入力: 会議で使用するため文房具を購入しました。事前に申請済みです。
出力:
購入物: 文房具
理由: 会議で使用するため
申請区分: 事前申請

例2:
入力: 社外セミナー参加費として交通費と宿泊費を申請します。
出力:
購入物: 交通費、宿泊費
理由: 社外セミナー参加
申請区分: 事前申請
"""
    prompt = few_shot + f"\n入力: {text}\n出力:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # =========================
    # few-shot例を削除
    output_text = output_text.replace(few_shot, "").strip()

    # =========================
    # 正規表現で「購入物」「理由」「申請区分」を抽出
    result = {}
    for key in ["購入物", "理由", "申請区分"]:
        match = re.search(f"{key}[:：]\s*(.*?)($|\n)", output_text)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = "不明"

    return result

# =========================
# 動作確認
sample_text = "昨日、実験で急に必要になったため研究用ケーブルを購入しました。事前に申請を行う時間が取れなかったため、事後での申請となります。"
print(expense_application_assistant(sample_text))
