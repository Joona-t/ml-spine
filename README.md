# ml-spine

12-week hands-on ML/RL learning spine. Every week ends in a from-scratch artifact,
built from memory after watching the source lecture — no code-alongs, no AI-written code
(see [CLAUDE.md](CLAUDE.md) for the tutor contract).

Hardware: Apple M1 Pro 16GB (MPS + MLX). Compute: local + free tiers only.

## Index

| Wk | Theme | Artifact | Key number | Status |
|----|-------|----------|------------|--------|
| 01 | Micrograd — backprop from scratch | — | — | ☐ |
| 02 | Bigram + MLP language models | — | — | ☐ |
| 03 | Backprop ninja + BatchNorm | — | — | ☐ |
| 04 | WaveNet + eigenvalues/PCA | — | — | ☐ |
| 05 | GPT from scratch | — | — | ☐ |
| 06 | BPE tokenizer (Finnish+English) | — | — | ☐ |
| 07 | REINFORCE — CartPole | — | — | ☐ |
| 08 | DQN — LunarLander | — | — | ☐ |
| 09 | PPO — LunarLander | — | — | ☐ |
| 10 | REINFORCE on my own char-GPT (KL penalty, reward hacking) | — | — | ☐ |
| 11 | DPO + GRPO from the papers | — | — | ☐ |
| 12 | MLX LoRA fine-tune capstone + retrospective | — | — | ☐ |

## Resources per week

| Wk | Watch/read |
|----|------------|
| 01 | Karpathy NN:Z2H L1; 3B1B Essence of Calculus ch.3–4 + ch.11; Khan multivariable partials |
| 02 | NN:Z2H L2–3; 3B1B Essence of Linear Algebra ch.4 + ch.9 |
| 03 | NN:Z2H L4–5 (the partials gap-closer — heaviest week, protect it) |
| 04 | NN:Z2H L6; 3B1B EoLA ch.14 (eigenvectors) |
| 05 | NN:Z2H L7; 3B1B "Attention in transformers"; karpathy/nanoGPT shakespeare-char |
| 06 | NN:Z2H L8; karpathy/minbpe exercise.md (do the exercise before reading his solution) |
| 07 | Spinning Up parts 1–3 (docs only); Sutton & Barto ch.3 + 13.1–13.4 |
| 08 | Sutton & Barto ch.6.1–6.5; CleanRL dqn.py only AFTER own attempt |
| 09 | Spinning Up PPO + GAE; "37 Implementation Details of PPO" (the debugging checklist) |
| 10 | HF "Illustrating RLHF" — then pure implementation on the W5 model |
| 11 | DPO paper (Rafailov 2023, full); DeepSeekMath §4 (GRPO) |
| 12 | mlx_lm.lora docs + mlx-examples/lora; MIT OCW 18.06 L29 (Strang, SVD) |

## Environment

```bash
brew install swig
uv sync                       # Python 3.12, torch (MPS), gymnasium[box2d], mlx, mlx-lm, trackio
# verification gates:
uv run python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
uv run python -c "import gymnasium as g; g.make('LunarLander-v3'); print('box2d OK')"
uv run python -c "import mlx.core as mx; print('MLX:', mx.default_device())"
uv run python -c "import trackio; print('trackio OK')"
```

Tracking: `import trackio as wandb; wandb.init(project="ml-spine")`, dashboard via `trackio show`.

## Rules

- One repo. Single-file scripts/notebooks. No frameworks, no meta-tools (urges → [IDEAS.md](IDEAS.md)).
- Week done = artifact runs + NOTES.md (≤500 words) + index row above + git tag `wNN`.
- If a week slips: CUT scope (cut list lives in the plan), never extend.
- Fallbacks pre-committed: W9 PPO stall → annotated own-PPO vs CleanRL bug-autopsy diff.
