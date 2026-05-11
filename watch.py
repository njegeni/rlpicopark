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

CHALLENGES = ["A_ascending", "B_stairs", "C_wall_drop", "D_gap", "E_tall_wall"]
SEEDS_PER_CHALLENGE = 3

results = []
with imageio.get_writer("trained_agent.mp4", fps=fps, macro_block_size=1) as writer:
    for challenge in CHALLENGES:
        method_name = f"_challenge_{challenge}"
        env.unwrapped._generate_obstacles = lambda mn=method_name: getattr(env.unwrapped, mn)()
        for seed in range(SEEDS_PER_CHALLENGE):
            obs, _ = env.reset(seed=seed)
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
            results.append((challenge, seed, outcome, ret, steps))
            print(f"{challenge:15} seed={seed} {outcome:7} return={ret:+.2f} steps={steps}")

env.close()

print()
print("=== Summary by challenge ===")
for challenge in CHALLENGES:
    rows = [r for r in results if r[0] == challenge]
    wins = sum(1 for r in rows if r[2] == "WIN")
    print(f"  {challenge:15} {wins}/{len(rows)} wins  mean_return={np.mean([r[3] for r in rows]):+.2f}")
