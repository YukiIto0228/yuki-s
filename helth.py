import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# === StepVerifier: 推論ステップごとの妥当性評価 ===
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
            return 0.3
        last_step = previous_steps[-1]
        overlap = len(set(last_step.split()) & set(candidate.split()))
        return min(max(overlap / (len(candidate.split()) + 1e-6), 0.3), 1.0)

    # 簡易妥当性スコア: 患者情報・症状との一致度
    def _symptom_score(self, patient_info, symptoms, candidate):
        count = sum(1 for s in symptoms.split("、") if s in candidate)
        return min(max(count / max(len(symptoms.split("、")), 1), 0.3), 1.0)

    def verify(self, task, previous_steps, candidate, patient_info="", symptoms=""):
        context = "\n".join(previous_steps) if previous_steps else "（なし）"
        text = f"""
課題:
{task}

これまでの推論:
{context}

次の推論ステップ候補:
{candidate}
"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        cls_score, cls_idx = torch.max(probs, dim=0)
        verdict = self.labels[cls_idx.item()]

        coherence_score = self._coherence_score(previous_steps, candidate)
        symptom_score = self._symptom_score(patient_info, symptoms, candidate)

        final_score = 0.5 * cls_score.item() + 0.3 * coherence_score + 0.2 * symptom_score
        reason = f"分類確率={cls_score:.2f}, 接続={coherence_score:.2f}, 症状一致={symptom_score:.2f}"

        return verdict, float(final_score), reason

# === StepGenerator: 次の推論ステップ候補を生成 ===
class StepGenerator:
    def __init__(self, model_name="rinna/japanese-gpt2-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, context, num_candidates=3):
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
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

# === ReasoningNode: 推論ステップとスコア管理 ===
class ReasoningNode:
    def __init__(self, steps=None, score=0.0):
        self.steps = steps if steps else []
        self.score = score

    def extend(self, step, step_score):
        return ReasoningNode(self.steps + [step], self.score + step_score)

# === 逐次症例推論 + 追加質問生成 ===
def sequential_medical_reasoning(patient_info, symptoms, generator, verifier, max_steps=7):
    node = ReasoningNode()
    facts = f"患者情報: {patient_info}\n症状: {symptoms}"
    task = "患者の症例を段階的に推測すること"

    for i in range(max_steps):
        context = f"課題: {task}\n観察結果:\n{facts}\nこれまでの推論:\n{' '.join(node.steps)}\n次の推論:"
        candidates = generator.generate(context, num_candidates=3)

        if not candidates:
            print("候補が生成されませんでした。終了します。")
            break

        # 各候補を評価
        scored_candidates = []
        for cand in candidates:
            verdict, score, reason = verifier.verify(task, node.steps, cand, patient_info, symptoms)
            scored_candidates.append((cand, verdict, score, reason))

        # 最も妥当な候補を選択
        best_step, best_verdict, best_score, best_reason = max(scored_candidates, key=lambda x: x[2])
        node = node.extend(best_step, best_score)

        print(f"\n=== ステップ {i+1} ===")
        print("選択ステップ:", best_step)
        print("ラベル:", best_verdict)
        print("スコア:", best_score)
        print("理由:", best_reason)

        # もし候補に追加質問が含まれる場合は、次ステップで入力として反映可能
        if "?" in best_step:
            response = input(f"{best_step}\n回答を入力してください: ")
            facts += f"\n追加情報: {best_step} → {response}"

    return node

# === 実行例 ===
if __name__ == "__main__":
    print("【患者情報（年齢・性別・基礎疾患など）を入力してください】")
    patient_info = input("> ")

    print("【症状を入力してください（例: 発熱、咳、腹痛）】")
    symptoms = input("> ")

    generator = StepGenerator()
    verifier = StepVerifier()

    final_node = sequential_medical_reasoning(patient_info, symptoms, generator, verifier, max_steps=7)

    print("\n=== 最終推測 ===")
    for i, step in enumerate(final_node.steps):
        print(f"ステップ {i+1}:", step)
    print("累積スコア:", final_node.score)
