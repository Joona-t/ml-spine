# BUGS_AND_ITERATIONS.md — ml-spine

Running log of every defect found + fix landed. Format: ID, date, problem, root cause, fix.

---

## ITER-001 (2026-06-11) — Repo bootstrap
- **What:** Scaffolded the 12-week learning spine: uv env (Python 3.12 — Homebrew 3.14 lacks box2d/pygame cp314 wheels), torch 2.12 (MPS verified), gymnasium[box2d] (swig installed via brew), mlx + mlx-lm, trackio. All four Phase 0 verification gates pass.
- **Why:** ML/RL pivot, dual-track plan (spine + sparkylab build track). Plan: ~/.claude/plans/squishy-riding-rainbow.md

## ITER-002 (2026-06-14) — Source links added
- **What:** Added verified links for Karpathy NN:Z2H, reference repos, math refreshers, RL/alignment sources, and MLX LoRA resources.
- **Why:** The scaffold named resources but did not link them, making the 1-hour/day study loop harder to start.
- **Fix:** Updated README source/resource tables only; no exercise code or scaffolding added.
