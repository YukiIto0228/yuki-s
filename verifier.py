from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


class StepVerifier:

    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small", device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        # NLIラベル対応
        self.id2label = self.model.config.id2label

    def verify(self, task, facts, steps, candidate):

        premise = (
            f"課題: {task}\n"
            f"観察結果: {facts}\n"
            f"これまでの考察: {' '.join(steps)}"
        )

        hypothesis = candidate

        # トークン化（NLIは2文入力）
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        # 予測ラベル
        pred_id = torch.argmax(probs).item()
        nli_label = self.id2label[pred_id]

        # NLIを評価ラベルへ変換
        verdict_map = {
            "entailment": ("妥当", 1.0),
            "neutral": ("根拠不足", 0.5),
            "contradiction": ("飛躍", 0.0)
        }

        verdict, score = verdict_map[nli_label]

        reason = (
            f"NLI判定: {nli_label} "
            f"(確率: {probs[pred_id]:.2f})"
        )

        return verdict, score, reason


# === テスト ===
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


