from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class StepVerifier:
    
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
    ).to(self.device)


    def verify(self, task, facts, steps, candidate):
        """
        ステップの妥当性を評価する
        
        Parameters
        ----------
        task: str
            課題文
        facts: str
            観察結果・事実
        steps: list[str]
            これまでの考察ステップ
        candidate: str
            検証対象ステップ
        
        Returns
        -------
        verdict: str
            評価ラベル（妥当 / 飛躍 / 根拠不足）
        score: float
            評価スコア（妥当=1.0, 根拠不足=0.5, 飛躍=0.0）
        reason: str
            簡単な理由
        """

        # プロンプト作成
        context = "\n".join(steps)
        prompt = f"""
課題:
{task}

観察結果:
{facts}

これまでの考察:
{context}

次の考察ステップ:
{candidate}

# 指示:
この考察ステップは妥当ですか？ 以下の3カテゴリのいずれかで答えてください:
- 妥当: 根拠が明確で論理的に妥当
- 飛躍: 根拠不足で論理的飛躍がある
- 根拠不足: 根拠が部分的に欠けている

形式は以下に従ってください：
ラベル: <妥当/飛躍/根拠不足>
スコア: <0.0〜1.0>
理由: <簡単な理由>
"""

        # トークン化
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False
        )

        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 簡易パース
        verdict, score, reason = self._parse_result(result_text)

        return verdict, score, reason

    def _parse_result(self, text):
        """
        出力テキストからラベル・スコア・理由を抽出
        """
        verdict, score, reason = "妥当", 1.0, "理由なし"
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("ラベル:"):
                verdict = line.split("ラベル:")[1].strip()
            elif line.startswith("スコア:"):
                try:
                    score = float(line.split("スコア:")[1].strip())
                except ValueError:
                    score = 1.0
            elif line.startswith("理由:"):
                reason = line.split("理由:")[1].strip()
        return verdict, score, reason

# === テスト実行 ===
if __name__ == "__main__":
    verifier = StepVerifier()
    task = "実験Aの結果を解釈し、考察を行う。"
    facts = "条件Xで結果Aが観察された。"
    steps = ["結果Aは条件Xに依存していると考えられる。"]
    candidate = "結果Aは過去研究Yと一致していると考えられる。"

    verdict, score, reason = verifier.verify(task, facts, steps, candidate)
    print("ラベル:", verdict)
    print("スコア:", score)
    print("理由:", reason)

