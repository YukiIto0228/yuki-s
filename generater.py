class StepGenerator:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate(self, context, num_candidates=3):
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=num_candidates,
            do_sample=True,
            temperature=0.7
        )
        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]