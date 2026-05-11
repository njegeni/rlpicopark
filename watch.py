import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3 import DQN

import PicoPark_v0  # registers PicoPark-v0


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(6)

    def action(self, a):
        return np.array([int(a) // 2, int(a) % 2], dtype=np.int64)


model = DQN.load("dqn_picopark")

env = DiscreteActionWrapper(gym.make("PicoPark-v0", render_mode="rgb_array"))
fps = env.metadata.get("render_fps", 10)

N_EPISODES = 15

results = []
with imageio.get_writer("trained_agent.mp4", fps=fps, macro_block_size=1) as writer:
    for seed in range(N_EPISODES):
        obs, _ = env.reset(seed=seed)
        layout = env.unwrapped._obstacles
        done = False
        ret = 0.0
        steps = 0
        writer.append_data(env.render())
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            steps += 1
            done = terminated or truncated
            writer.append_data(env.render())

        if terminated and ret > 0.5:
            outcome = "WIN"
        elif terminated:
            outcome = "DIED"
        else:
            outcome = "TIMEOUT"
        n_obs = len(layout)
        results.append((seed, outcome, ret, steps, n_obs))
        print(f"seed={seed:2}  {outcome:7}  return={ret:+.2f}  steps={steps:3}  obstacles={n_obs}")

env.close()

print()
wins = sum(1 for r in results if r[1] == "WIN")
print(f"=== {wins}/{len(results)} wins ({100*wins/len(results):.0f}%) ===")
print(f"mean return: {np.mean([r[2] for r in results]):+.2f}")
