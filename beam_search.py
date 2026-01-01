class ReasoningNode:

    def __init__(self, steps, score, parent=None):
        self.steps = steps      # 推論ステップのリスト
        self.score = score      # 累積スコア
        self.parent = parent    # 親ノード

    def count(self):
        return len(self.steps)

    def extend(self, new_step, step_score):
        return ReasoningNode(
            self.steps + [new_step],
            self.score + step_score,
            self
        )


def beam_search(
    x,                   # 問題文
    beam_size,           # ビーム幅
    max_depth,           # 最大ステップ数
    M,                   # 各ノードから出す候補数
    generate_next_steps, # 次ステップ生成関数
    verify_step          # 検証器
):
    # 初期状態（空の推論）
    beam = [ReasoningNode([], 0.0)]

    # 推論ステップを1段ずつ進める
    for _ in range(max_depth):
        candidates = []

        for node in beam:
            # 次の推論ステップ候補を生成
            next_steps = generate_next_steps(
                x,
                node.steps,
                M
            )

            for step in next_steps:
                # ステップを検証
                step_score = verify_step(
                    x,
                    node.steps,
                    step
                )

                # 新しいノードを作る
                new_node = node.extend(step, step_score)
                candidates.append(new_node)

        # スコア順に並べて上位だけ残す（枝刈り）
        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_size]

        if len(beam) == 0:
            break

    return beam
