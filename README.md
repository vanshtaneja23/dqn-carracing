# DQN CarRacing

A Deep Q-Network agent for Gymnasium **CarRacing-v3**, learning to drive from pixels with 5 discrete
actions, built in PyTorch. A small Flask page shows the measured results.

## Status

Training on the real environment is in progress. **No results are published yet.** The numbers this
README used to show (847 average reward, 92% lap completion) were not real and have been removed.
"What went wrong" below explains why.

## What went wrong, and what I fixed

- **The committed training run never saw CarRacing.** When `CarRacing-v2` couldn't be created, the
  environment wrapper silently fell back to a mock environment that returns random frames and random
  rewards. Gymnasium 1.x renamed the environment to `CarRacing-v3`, so the fallback always fired. The
  evidence: all 2,000 logged episodes were exactly 251 steps long, and a random policy on the mock
  averages about 898, the same as the "trained" agent.
- **Fixed:** the wrapper now uses `CarRacing-v3` and raises an error with an install hint instead of
  mocking. The mock remains only behind an explicit `use_mock=True` for unit tests, `is_mock` is
  recorded in each run's config, and `test_setup.py` asserts training runs on the real environment.
  The invalid runs were deleted.
- **Bugs that only showed up on the real environment:**
  - Actions were passed as Python lists, which crashes CarRacing-v3. They are now NumPy arrays.
  - Epsilon decayed on every gradient step, so exploration hit its floor after about 4 episodes.
    It now decays once per episode.
  - The device setting ignored Apple Silicon. It now picks CUDA, MPS or CPU.
  - MSE loss on large shaped rewards. It now uses Huber loss, the standard choice for DQN.
- **Results are reported only from measurement.** `evaluate_agent.py` scores the agent on the standard
  (unshaped) reward and writes `eval.json`; training writes per-episode raw and shaped rewards to
  `rewards.json`. The web page reads only those files.

## How it works

| Part | Choice |
|---|---|
| Input | 4 stacked 84×84 grayscale frames, 4-frame skip |
| Network | 3 conv layers (32, 64, 64 filters) → 512 fully connected → 5 actions |
| Learning | Experience replay (100K transitions), target network synced every 1,000 updates, Adam (lr 1e-4), Huber loss |
| Exploration | ε-greedy from 1.0 to 0.01, decayed ×0.995 per episode |
| Training-only shaping | Boosted on-track rewards, heavier penalties for long off-track stretches, early stop when stuck |
| Evaluation | Standard CarRacing-v3 reward, no shaping, 20 episodes. The environment counts as solved at 900+ |

## Run it

```bash
pip install -r requirements-train.txt
python test_setup.py                  # checks the real environment loads
python training.py --episodes 1000    # writes runs/<run>/ (checkpoints, TensorBoard logs, rewards.json)
python evaluate_agent.py              # 20 episodes on the standard reward -> runs/<run>/eval.json
```

The results page: `pip install -r requirements.txt && python app.py`, then open http://localhost:8080.

## License

MIT
