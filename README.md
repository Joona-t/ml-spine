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

## Sources

Primary spine: [Karpathy Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) + from-memory rebuilds.
Reference repos: [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero), [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), [minBPE](https://github.com/karpathy/minbpe), [nanoGPT](https://github.com/karpathy/nanoGPT).

## Resources per week

| Wk | Watch/read |
|----|------------|
| 01 | [Karpathy NN:Z2H L1 — micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0); [3B1B Essence of Calculus](https://www.3blue1brown.com/topics/calculus) ch.3–4 + ch.11; [Khan multivariable derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) |
| 02 | [NN:Z2H L2 — bigram language model](https://www.youtube.com/watch?v=PaCmpygFfXo); [NN:Z2H L3 — MLP makemore](https://youtu.be/TCH_1BHY58I); [3B1B Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) ch.4 + ch.9 |
| 03 | [NN:Z2H L4 — activations, gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc); [NN:Z2H L5 — backprop ninja](https://youtu.be/q8SA3rM6ckI) (the partials gap-closer — heaviest week, protect it) |
| 04 | [NN:Z2H L6 — WaveNet](https://youtu.be/t3YJ5hKiMQ0); [3B1B EoLA eigenvectors/eigenvalues](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 05 | [NN:Z2H L7 — GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY); [3B1B Attention in transformers](https://www.3blue1brown.com/lessons/attention); [nanoGPT](https://github.com/karpathy/nanoGPT) shakespeare-char |
| 06 | [NN:Z2H L8 — GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE); [minBPE](https://github.com/karpathy/minbpe) exercise.md (do the exercise before reading his solution) |
| 07 | [Spinning Up RL intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) parts 1–3 (docs only); [Sutton & Barto RL book](http://incompleteideas.net/book/the-book-2nd.html) ch.3 + 13.1–13.4 |
| 08 | [Sutton & Barto RL book PDF](http://incompleteideas.net/book/RLbook2020.pdf) ch.6.1–6.5; [CleanRL dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) only AFTER own attempt |
| 09 | [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) + GAE; [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) debugging checklist |
| 10 | [Hugging Face RLHF overview](https://huggingface.co/blog/rlhf) — then pure implementation on the W5 model |
| 11 | [DPO paper — Rafailov et al. 2023](https://arxiv.org/abs/2305.18290); [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) §4 (GRPO) |
| 12 | [MLX LM fine-tuning docs](https://github.com/ml-explore/mlx-lm/blob/main/README.md#fine-tuning) + [MLX LoRA examples](https://github.com/ml-explore/mlx-examples/tree/main/lora); [MIT OCW 18.06 L29 — SVD](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/) |

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
