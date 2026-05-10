import pickle
import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt

import PicoPark_v0


n_actions = 6  # 3 horizontal directions × 2 jump options


def obs_to_state(obs):
    agent_x = int(obs["agent"][0])
    agent_y = int(obs["agent"][1])
    vy = int(obs["vy"][0])
    return (agent_x, agent_y, vy)


def action_to_env(a):
    h_dir = a // 2
    jump = a % 2
    return np.array([h_dir, jump], dtype=np.int64)


# agent

class PicoParkAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(n_actions)
        else:
            return int(np.argmax(self.q_values[state]))

    def update(self, state, action, reward, terminated, next_state):
        future_q = (not terminated) * np.max(self.q_values[next_state])
        td_target = reward + self.discount_factor * future_q
        td_error = td_target - self.q_values[state][action]

        self.q_values[state][action] += self.lr * td_error

        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


#hyperparameters

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make('PicoPark-v0')

agent = PicoParkAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

episode_returns = []

for episode in tqdm(range(n_episodes)):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    state = obs_to_state(obs)
    done = False
    episode_return = 0.0

    # Run one episode
    while not done:
        #Agent chooses action with respect to the epsilon decay (exploration -> exploitation )
        action = agent.get_action(state)

        #Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action_to_env(action))
        next_state = obs_to_state(next_obs)

        #Learn from this experience
        agent.update(state, action, reward, terminated, next_state)

        #Move to next state
        done = terminated or truncated
        state = next_state
        episode_return += reward

    episode_returns.append(episode_return)

    #must reduce exploration rate over time
    agent.decay_epsilon()


# --- diagnostics ---

def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


window = max(1, n_episodes // 100)
returns_smoothed = rolling_mean(episode_returns, window)
td_smoothed = rolling_mean(agent.training_error, max(1, len(agent.training_error) // 100))

n_first = max(1, n_episodes // 20)
print(f"states visited: {len(agent.q_values)}")
print(f"mean return  first {n_first}: {np.mean(episode_returns[:n_first]):.3f}")
print(f"mean return   last {n_first}: {np.mean(episode_returns[-n_first:]):.3f}")
print(f"mean |TD err| first {n_first}: {np.mean(np.abs(agent.training_error[:n_first])):.4f}")
print(f"mean |TD err|  last {n_first}: {np.mean(np.abs(agent.training_error[-n_first:])):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(returns_smoothed)
axes[0].set_title(f"episode return (rolling mean, w={window})")
axes[0].set_xlabel("episode")
axes[1].plot(td_smoothed)
axes[1].set_title("|TD error| (rolling mean)")
axes[1].set_xlabel("update step")
plt.tight_layout()
plt.savefig("training_diagnostics.png", dpi=120)
print("saved training_diagnostics.png")

# greedy eval
agent.epsilon = 0.0
eval_returns = []
for _ in range(20):
    obs, _ = env.reset()
    state = obs_to_state(obs)
    done = False
    ret = 0.0
    while not done:
        action = agent.get_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action_to_env(action))
        state = obs_to_state(next_obs)
        ret += reward
        done = terminated or truncated
    eval_returns.append(ret)

print(f"greedy eval over 20 eps: mean={np.mean(eval_returns):.3f}  std={np.std(eval_returns):.3f}  min={np.min(eval_returns):.3f}  max={np.max(eval_returns):.3f}")


# save Q-table for later use (watch.py loads this)
q_table_path = "q_table.pkl"
with open(q_table_path, "wb") as f:
    pickle.dump(dict(agent.q_values), f)
print(f"saved {q_table_path} ({len(agent.q_values)} states)")
