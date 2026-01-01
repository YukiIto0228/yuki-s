import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

class StepVerifier:

    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small", device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.id2label = self.model.config.id2label

    def verify(self, task, facts, steps, candidate):
        premise = (
            f"課題: {task}\n"
            f"観察結果: {facts}\n"
            f"これまでの考察: {' '.join(steps)}"
        )

        hypothesis = candidate

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        pred_id = torch.argmax(probs).item()
        label = self.id2label[pred_id]

        verdict_map = {
            "entailment": 1.0,
            "neutral": 0.5,
            "contradiction": 0.0
        }

        return verdict_map[label]


class StepGenerator:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, context, num_candidates=3):
        inputs = self.tokenizer(
            context,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7
        )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]


class ReasoningNode:
    def __init__(self, steps, score, parent=None):
        self.steps = steps
        self.score = score
        self.parent = parent

    def extend(self, new_step, step_score):
        return ReasoningNode(
            self.steps + [new_step],
            self.score + step_score,
            self
        )


def beam_search(
    task,
    facts,
    beam_size,
    max_depth,
    M,
    generator,
    verifier
):
    beam = [ReasoningNode([], 0.0)]

    for depth in range(max_depth):
        print(f"\n=== Depth {depth + 1} ===")
        candidates = []

        for node in beam:
            context = (
                f"課題: {task}\n"
                f"観察結果: {facts}\n"
                f"これまでの考察: {' '.join(node.steps)}\n"
                f"次の考察:"
            )

            next_steps = generator.generate(context, M)

            for step in next_steps:
                step_score = verifier.verify(
                    task,
                    facts,
                    node.steps,
                    step
                )

                new_node = node.extend(step, step_score)
                candidates.append(new_node)

                print("候補:", step[:60].replace("\n", " "))
                print("  スコア:", step_score)

        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_size]

        if not beam:
            break

    return beam


if __name__ == "__main__":
    task = "実験Aの結果を解釈し、考察を行う。"
    facts = "条件Xで結果Aが観察された。"

    generator = StepGenerator()
    verifier = StepVerifier()

    result = beam_search(
        task=task,
        facts=facts,
        beam_size=2,
        max_depth=2,
        M=3,
        generator=generator,
        verifier=verifier
    )

    print("\n=== 最終結果 ===")
    for i, node in enumerate(result):
        print(f"\n[候補 {i+1}] 累積スコア: {node.score}")
        for s in node.steps:
            print("-", s)
