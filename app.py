from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# ===== モデル設定（軽量で認証不要の日本語モデル） =====
model_name = "rinna/japanese-gpt-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPUがあれば自動で割り当て
    torch_dtype=torch.float16,  # 可能なら軽量化
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
)

# ===== モデル出力の整理関数 =====
def parse_expense_output(output_text):
    """
    モデル出力から「購入物」「理由」「申請区分」を抽出。
    複数の入力文がある場合はそれぞれの出力を辞書に分ける。
    """
    results = []

    # 「入力：〜 出力：」ごとに分割
    pattern = r"入力[:：](.+?)\s*出力[:：](.+?)(?=(?:\n入力[:：])|$)"
    matches = re.findall(pattern, output_text, re.DOTALL)

    for input_text, out_text in matches:
        entry = {}
        for key in ["購入物", "理由", "申請区分"]:
            match = re.search(rf"{key}[:：]\s*(.*?)(?:\n|$)", out_text)
            entry[key] = match.group(1).strip() if match else "不明"
        entry["入力文"] = input_text.strip()
        results.append(entry)

    return results

# ===== テスト用入力 =====
texts = [
    "会議で使用するため文房具を購入しました。事前に申請済みです。",
    "実験用ケーブルを購入しました。急ぎのため事後申請です。",
]

# ===== モデルに入力して出力取得 =====
prompt_template = """以下の文章から「購入物」「理由」「申請区分」を抽出してください。
入力: {text}
出力:"""

all_output = ""
for t in texts:
    prompt = prompt_template.format(text=t)
    out = generator(prompt, max_new_tokens=128)[0]["generated_text"]
    all_output += out + "\n"

# ===== 整理して表示 =====
parsed = parse_expense_output(all_output)
for p in parsed:
    print(p)

