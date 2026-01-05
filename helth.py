import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# -----------------------------
# StepVerifier: 経費妥当性検証器
# -----------------------------
class StepVerifier:
    """
    経費ステップの妥当性を複数指標で評価
    """
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-v3", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        ).to(self.device)
        self.labels = ["妥当", "根拠不足", "飛躍"]

        # 社内ルール
        self.RULES = {
            "交通費": {"上限": 10000, "許可用途": ["出張", "営業訪問"]},
            "交際費": {"上限": 20000, "許可用途": ["接待", "会食"]},
            "備品費": {"上限": 50000, "許可用途": ["事務用品", "PC購入", "PC周辺機器"]},
        }

    # 数値整合性スコア
    def _numeric_score(self, step_text):
        numbers = list(map(int, re.findall(r"\d+", step_text)))
        if not numbers:
            return 0.3
        return 0.6 if len(numbers) <= 3 else 0.9

    # 理由妥当性スコア
    def _reason_score(self, step_text):
        for item, rule in self.RULES.items():
            if item in step_text:
                if any(u in step_text for u in rule["許可用途"]):
                    return 1.0
                else:
                    return 0.3
        return 0.5  # 不明項目は中間評価

    # 社内規程スコア（上限チェック）
    def _rule_score(self, step_text):
        for item, rule in self.RULES.items():
            if item in step_text:
                amounts = list(map(int, re.findall(r"\d+", step_text)))
                if amounts and amounts[0] > rule["上限"]:
                    return 0.0
                else:
                    return 1.0
        return 0.5

    # LLM分類スコア
    def _llm_score(self, step_text):
        inputs = self.tokenizer(step_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
            cls_score, cls_idx = torch.max(probs, dim=0)
        return cls_score.item(), self.labels[cls_idx.item()]

    # 総合評価
    def verify(self, step_text):
        num_score = self._numeric_score(step_text)
        reason_score = self._reason_score(step_text)
        rule_score = self._rule_score(step_text)
        llm_score, llm_label = self._llm_score(step_text)

        # 重み付き統合
        final_score = 0.3 * num_score + 0.3 * reason_score + 0.2 * rule_score + 0.2 * llm_score

        # 最終判定
        if final_score >= 0.75:
            verdict = "妥当"
        elif final_score >= 0.5:
            verdict = "根拠不足"
        else:
            verdict = "飛躍"

        scores = {
            "numeric": num_score,
            "reason": reason_score,
            "rule": rule_score,
            "llm": llm_score,
            "final": final_score
        }

        return verdict, scores

# -----------------------------
# StepGenerator
# -----------------------------
class StepGenerator:
    def __init__(self, model_name="rinna/japanese‑gpt‑neox‑3.6b", device=None):
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
            max_new_tokens=40,
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

# -----------------------------
# Reasoning Node & Beam Search
# -----------------------------
class ReasoningNode:
    def __init__(self, steps, score, parent=None):
        self.steps = steps
        self.score = score
        self.parent = parent

    def extend(self, step, step_score):
        return ReasoningNode(self.steps + [step], self.score + step_score, parent=self)

def beam_search(task, facts, beam_size, max_depth, M, generator, verifier):
    beam = [ReasoningNode([], 0.0)]
    for depth in range(max_depth):
        print(f"\n=== Depth {depth + 1} ===")
        candidates = []
        for node in beam:
            context = f"課題: {task}\n観察結果: {facts}\nこれまでの考察: {' '.join(node.steps)}\n次の経費ステップ:"
            steps = generator.generate(context, M)
            for step in steps:
                verdict, scores = verifier.verify(step)
                final_score = scores["final"]
                new_node = node.extend(step, final_score)
                candidates.append(new_node)
                print(f"候補: {step[:60].replace(chr(10),' ')}")
                print(f"  判定: {verdict}, 総合スコア: {final_score:.2f}")
                print(f"  指標スコア: {scores}")
        # Beam pruning
        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_size]
        if not beam:
            break
    return beam

# -----------------------------
# 実行例
# -----------------------------
if __name__ == "__main__":
    print("【観察結果・事実を入力】")
    facts = input("> ")

    print("\n【課題（経費妥当性の判定など）を入力】")
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

    print("\n=== 最終結果（候補パス） ===")
    for i, node in enumerate(results):
        print(f"\n[候補 {i+1}] 累積スコア: {node.score:.2f}")
        for step in node.steps:
            print("-", step)


