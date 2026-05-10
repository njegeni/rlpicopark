import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class PicoParkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=15):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = 200 # Maximum steps to ensure episodes end

        # Platformer physics
        self.ground_y = 2
        self.JUMP_V0 = 2   # initial vertical velocity on jump
        self.GRAVITY = 1   # vy decreases by this each step

        # Observations: agent + target positions, plus current vertical velocity
        # so the policy knows whether it is airborne and committed to an arc.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "vy": spaces.Box(-3, self.JUMP_V0, shape=(1,), dtype=np.int32),
            }
        )

        # MultiDiscrete: dim 0 is horizontal (0=left, 1=none, 2=right),
        # dim 1 is the jump button (0=no, 1=yes). They are independent each step,
        # so the agent can hold a direction while pressing jump.
        self.action_space = spaces.MultiDiscrete([3, 2])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "vy": np.array([self._vy], dtype=np.int32),
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.current_step = 0

        # Agent starts at random x near the left, on the ground; door (target) at far right.
        start_x = self.np_random.integers(0, self.size // 4)
        self._agent_location = np.array([start_x, self.ground_y], dtype=np.int32)
        self._target_location = np.array([self.size - 2, self.ground_y], dtype=np.int32)

        # Physics state
        self._vy = 0          # vertical velocity

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1
        prev_location = self._agent_location.copy()

        h_dir = int(action[0])      # 0=left, 1=none, 2=right
        jump_btn = int(action[1])   # 0/1
        dx = h_dir - 1              # -> -1, 0, +1

        # Position is the source of truth: jumps are only allowed when standing on the ground.
        grounded = self._agent_location[1] == self.ground_y
        if grounded and jump_btn:
            self._vy = self.JUMP_V0

        # Apply kinematics: x_{t+1} = x_t + dx,  y_{t+1} = y_t + vy_t,  vy_{t+1} = vy_t - g
        self._agent_location[0] += dx
        self._agent_location[1] += self._vy
        self._vy -= self.GRAVITY

        # Land on the ground: clamp y and zero out vertical velocity.
        if self._agent_location[1] <= self.ground_y:
            self._agent_location[1] = self.ground_y
            self._vy = 0

        # Stay inside the grid horizontally.
        self._agent_location[0] = np.clip(self._agent_location[0], 0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = self.current_step >= self.max_steps

        # Reward function: shaped progress + small step cost + terminal bonus
        prev_distance = np.linalg.norm(prev_location - self._target_location, ord=1)
        curr_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)

        reward = 0.1 * (prev_distance - curr_distance) - 0.01
        if terminated:
            reward += 1.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size 
        )  # The size of a single grid square in pixels

        # Flip y for screen coords: in pygame, y=0 is the top, but our world has y=0 at the bottom.
        def to_screen(pos):
            return np.array([pos[0], self.size - 1 - pos[1]], dtype=np.float64)

        # Floor block: rows game-y in [0, ground_y - 1] (the agent stands on top of it, on game-y=ground_y).
        floor_top_row = self.size - self.ground_y   # screen-row index of the floor's top edge
        floor_top_px = floor_top_row * pix_square_size
        pygame.draw.rect(
            canvas,
            (255, 140, 75),  # pico park orange
            pygame.Rect(0, floor_top_px, self.window_size, self.window_size - floor_top_px),
        )

        # Gridlines only above the floor, drawn light grey. Drawn before the door/agent
        # so those shapes naturally cover any line passing through them.
        grid_color = (200, 200, 200)
        for row in range(floor_top_row + 1):
            y = pix_square_size * row
            pygame.draw.line(canvas, grid_color, (0, y), (self.window_size, y), width=1)
        for col in range(self.size + 1):
            x = pix_square_size * col
            pygame.draw.line(canvas, grid_color, (x, 0), (x, floor_top_px), width=1)

        # Draw the target as a 2-tall door (occupies game-y in [ground_y, ground_y + 1]).
        door_top = to_screen(self._target_location + np.array([0, 1])) * pix_square_size
        door_rect = pygame.Rect(door_top[0], door_top[1], pix_square_size, 2 * pix_square_size)
        pygame.draw.rect(canvas, (65, 25, 0), door_rect)
        # Door knob
        knob_radius = max(2, int(pix_square_size * 0.08))
        pygame.draw.circle(
            canvas,
            (230, 200, 90),
            (door_rect.right - knob_radius * 2, door_rect.centery),
            knob_radius,
        )

        # Draw the agent as a Pico Park-style character: rounded square body + two eyes.
        agent_center = (to_screen(self._agent_location) + 0.5) * pix_square_size
        body_size = pix_square_size * 0.75
        body_rect = pygame.Rect(
            agent_center[0] - body_size / 2,
            agent_center[1] - body_size / 2,
            body_size,
            body_size,
        )
        pygame.draw.rect(canvas, (50, 150, 220), body_rect, border_radius=int(body_size * 0.3))
        # Eyes
        eye_dx = body_size * 0.2
        eye_dy = body_size * 0.1
        eye_radius = max(2, int(body_size * 0.1))
        pygame.draw.circle(canvas, (255, 255, 255),
                           (int(agent_center[0] - eye_dx), int(agent_center[1] - eye_dy)),
                           eye_radius + 1)
        pygame.draw.circle(canvas, (255, 255, 255),
                           (int(agent_center[0] + eye_dx), int(agent_center[1] - eye_dy)),
                           eye_radius + 1)
        pygame.draw.circle(canvas, (0, 0, 0),
                           (int(agent_center[0] - eye_dx), int(agent_center[1] - eye_dy)),
                           eye_radius)
        pygame.draw.circle(canvas, (0, 0, 0),
                           (int(agent_center[0] + eye_dx), int(agent_center[1] - eye_dy)),
                           eye_radius)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
