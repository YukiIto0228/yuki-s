import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

class StepVerifier:
    """
    Classification-based step verifier + numeric heuristic
    """

    def __init__(
        self,
        model_name="cl-tohoku/bert-base-japanese-v3",
        device=None
    ):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3   # 妥当 / 根拠不足 / 飛躍
        ).to(self.device)

        self.labels = ["妥当", "根拠不足", "飛躍"]

    # -------------------------
    # 数値ヒューリスティック
    # -------------------------
    def _numeric_consistency(self, facts, steps, candidate):
        numbers = list(map(int, re.findall(r"\d+", facts + candidate)))
        if not numbers:
            return 0.3  # 数字が無い推論は弱い

        return 0.6 if len(numbers) <= 3 else 0.9

    # -------------------------
    # Verifier 本体
    # -------------------------
    def verify(self, task, facts, steps, candidate):

        context = "\n".join(steps) if steps else "（なし）"

        text = f"""
課題:
{task}

観察結果:
{facts}

これまでの考察:
{context}

次の考察ステップ:
{candidate}
"""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        cls_score, cls_idx = torch.max(probs, dim=0)
        verdict = self.labels[cls_idx.item()]

        # ヒューリスティックと融合
        num_score = self._numeric_consistency(facts, steps, candidate)

        final_score = (
            0.7 * cls_score.item() +
            0.3 * num_score
        )

        reason = f"分類確率={cls_score:.2f}, 数値整合性={num_score:.2f}"

        return verdict, float(final_score), reason

class StepGenerator:
  
    def __init__(self, model_name="rinna/japanese-gpt2-small", device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, context, num_candidates):
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        steps = []
        for seq in outputs:
            step = self.tokenizer.decode(
                seq[input_len:], skip_special_tokens=True
            ).strip()
            if step:
                steps.append(step)

        return steps


# =========================
# Reasoning Node
# =========================
class ReasoningNode:
    def __init__(self, steps, score, parent=None):
        self.steps = steps
        self.score = score
        self.parent = parent

    def extend(self, step, step_score):
        return ReasoningNode(
            self.steps + [step],
            self.score + step_score,
            parent=self
        )


# =========================
# Beam Search over Steps
# =========================
def beam_search(task, facts, beam_size, max_depth, M, generator, verifier):
    beam = [ReasoningNode([], 0.0)]

    for depth in range(max_depth):
        print(f"\n=== Depth {depth + 1} ===")
        candidates = []

        for node in beam:
            context = (
                f"課題: {task}\n"
                f"観察結果: {facts}\n"
                f"これまでの考察:\n"
                f"{' '.join(node.steps)}\n"
                f"次の考察:"
            )

            # 理論どおり：常に M 個生成
            steps = generator.generate(context, M)

            for step in steps:
                verdict, step_score, reason = verifier.verify(
                    task, facts, node.steps, step
                )

                new_node = node.extend(step, step_score)
                candidates.append(new_node)

                print("候補:", step[:60].replace("\n", " "))
                print("  ラベル:", verdict, "スコア:", step_score)

        # Beam pruning
        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_size]

        if not beam:
            break

    return beam


if __name__ == "__main__":

    print("【元の文章（観察結果・事実）を入力してください】")
    facts = input("> ")

    print("\n【課題（どういう推論・出力をしたいか）を入力してください】")
    task = input("> ")

    generator = StepGenerator()
    verifier = StepVerifier()

    results = beam_search(
        task=task,
        facts=facts,
        beam_size=2,
        max_depth=2,
        M=3,
        generator=generator,
        verifier=verifier
    )

    print("\n=== 最終結果 ===")
    for i, node in enumerate(results):
        print(f"\n[候補 {i+1}] 累積スコア: {node.score:.2f}")
        for step in node.steps:
            print("-", step)

