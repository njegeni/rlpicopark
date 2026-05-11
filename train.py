import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3 import DQN

import PicoPark_v0  # registers PicoPark-v0


class DiscreteActionWrapper(gym.ActionWrapper):
    """SB3 DQN needs a Discrete action space; the env uses MultiDiscrete([3, 2]).
    Action a in {0..5}: a // 2 = h_dir (0=left, 1=none, 2=right), a % 2 = jump."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(6)

    def action(self, a):
        return np.array([int(a) // 2, int(a) % 2], dtype=np.int64)


env = DiscreteActionWrapper(gym.make("PicoPark-v0"))

model = DQN(
    "MultiInputPolicy",   # handles the Dict observation space
    env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.95,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.5,   # epsilon decays over first 50% of training
    exploration_final_eps=0.05,
    policy_kwargs={"net_arch": [64, 64]},
    verbose=0,
)

total_timesteps = 500_000
model.learn(total_timesteps=total_timesteps, progress_bar=True)

model.save("dqn_picopark")
print("saved dqn_picopark.zip")

# Greedy eval
returns = []
for _ in range(20):
    obs, _ = env.reset()
    done = False
    ret = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ret += r
        done = term or trunc
    returns.append(ret)

print(f"greedy eval over 20 eps: "
      f"mean={np.mean(returns):.3f}  std={np.std(returns):.3f}  "
      f"min={np.min(returns):.3f}  max={np.max(returns):.3f}")
