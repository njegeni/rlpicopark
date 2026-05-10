import pickle
from collections import defaultdict

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

import PicoPark_v0  # registers PicoPark-v0


n_actions = 6


def obs_to_state(obs):
    return (int(obs["agent"][0]), int(obs["agent"][1]), int(obs["vy"][0]))


def action_to_env(a):
    return np.array([a // 2, a % 2], dtype=np.int64)


with open("q_table.pkl", "rb") as f:
    q = defaultdict(lambda: np.zeros(n_actions), pickle.load(f))

env = gym.make("PicoPark-v0", render_mode="rgb_array")
fps = env.metadata.get("render_fps", 10)

with imageio.get_writer("trained_agent.mp4", fps=fps, macro_block_size=1) as writer:
    for ep in range(5):
        obs, _ = env.reset()
        state = obs_to_state(obs)
        done = False
        ret = 0.0
        writer.append_data(env.render())
        while not done:
            action = int(np.argmax(q[state]))
            next_obs, reward, terminated, truncated, _ = env.step(action_to_env(action))
            state = obs_to_state(next_obs)
            ret += reward
            done = terminated or truncated
            writer.append_data(env.render())
        print(f"episode {ep + 1}: return={ret:.3f}")

env.close()
