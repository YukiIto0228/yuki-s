import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# === StepVerifier: 各推論ステップを評価 ===
class StepVerifier:
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-v3", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # 妥当 / 根拠不足 / 飛躍
        ).to(self.device)
        self.labels = ["妥当", "根拠不足", "飛躍"]

    # 文脈との整合性スコア
    def _coherence_score(self, previous_steps, candidate):
        if not previous_steps:
            return 0.5
        last_step = previous_steps[-1]
        prev_words = set(last_step.split())
        cand_words = set(candidate.split())
        overlap = len(prev_words & cand_words)
        return min(max(overlap / (len(cand_words) + 1e-6), 0.3), 1.0)

    # 推論ステップの妥当性評価
    def verify(self, task, steps, candidate):
        context = "\n".join(steps) if steps else "（なし）"
        text = f"""
課題:
{task}

これまでの推論:
{context}

次の推論ステップ候補:
{candidate}
"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        cls_score, cls_idx = torch.max(probs, dim=0)
        verdict = self.labels[cls_idx.item()]

        coherence_score = self._coherence_score(steps, candidate)
        final_score = 0.7 * cls_score.item() + 0.3 * coherence_score
        reason = f"分類確率={cls_score:.2f}, 接続={coherence_score:.2f}"

        return verdict, float(final_score), reason

# === StepGenerator: 次の症例候補を生成 ===
class StepGenerator:
    def __init__(self, model_name="rinna/japanese-gpt2-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, context, num_candidates=3, max_new_tokens=50):
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 修正
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        steps = []
        for seq in outputs:
            step = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            if step:
                steps.append(step)
        return steps

# === ReasoningNode: 推論ステップとスコアを管理 ===
class ReasoningNode:
    def __init__(self, steps=None, score=0.0):
        self.steps = steps if steps else []
        self.score = score

    def extend(self, step, step_score):
        return ReasoningNode(self.steps + [step], self.score + step_score)

# === 逐次ステップ推論（症例推測） ===
def sequential_reasoning(task, patient_info, symptoms, generator, verifier, max_steps=5):
    node = ReasoningNode()
    facts = f"患者情報: {patient_info}\n症状: {symptoms}"

    for i in range(max_steps):
        context = f"課題: {task}\n観察結果:\n{facts}\nこれまでの推論:\n{' '.join(node.steps)}\n次の推論:"
        candidates = generator.generate(context, num_candidates=3)

        if not candidates:
            print("候補が生成されませんでした。終了します。")
            break

        # 各候補を評価
        scored_candidates = []
        for cand in candidates:
            verdict, score, reason = verifier.verify(task, node.steps, cand)
            scored_candidates.append((cand, verdict, score, reason))

        # 最もスコアの高い候補を選択
        best_step, best_verdict, best_score, best_reason = max(scored_candidates, key=lambda x: x[2])
        node = node.extend(best_step, best_score)

        print(f"\n=== ステップ {i+1} ===")
        print("選択ステップ:", best_step)
        print("ラベル:", best_verdict)
        print("スコア:", best_score)
        print("理由:", best_reason)

    return node

# === 実行例 ===
if __name__ == "__main__":
    print("【患者情報（年齢・性別・基礎疾患など）を入力してください】")
    patient_info = input("> ")

    print("【症状を入力してください（例: 発熱、咳、腹痛）】")
    symptoms = input("> ")

    task = "患者の症例を段階的に推測すること（風邪や日常症状に限定）"

    generator = StepGenerator()
    verifier = StepVerifier()

    final_node = sequential_reasoning(task, patient_info, symptoms, generator, verifier, max_steps=5)

    print("\n=== 最終推測 ===")
    for i, step in enumerate(final_node.steps):
        print(f"ステップ {i+1}:", step)
    print("累積スコア:", final_node.score)

