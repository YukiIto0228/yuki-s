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
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        context_len = inputs["input_ids"].shape[1]

        # 【変更①】
        # context 全体ではなく「新しく生成された部分のみ」を step として返す
        steps = [
            self.tokenizer.decode(seq[context_len:], skip_special_tokens=True).strip()
            for seq in outputs.sequences
        ]

        # 尤度（簡易：最終トークンの平均 logit）
        likelihoods = [
            torch.mean(torch.stack([s.mean() for s in outputs.scores])).item()
            for _ in steps
        ]

        return list(zip(steps, likelihoods))
