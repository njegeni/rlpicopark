import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


# Each obstacle is a tuple (kind, x, w, h):
#   kind = "platform" -> solid block at x..x+w-1, h tiles tall above ground (landable on top)
#   kind = "pit"      -> gap in floor at x..x+w-1 (h unused)
#
# Physics reminders (size=15, ground_y=2, JUMP_V0=2, GRAVITY=2):
#   - max jump = +2 above the y the agent starts from (ground or platform top)
#   - from ground (y=2): can land on h<=2, blocked horizontally by h>=3
#   - from h=1 step (y=3): can land on h<=3; from h=3 (y=5): can land on h<=5; etc.
#   - stair sequences with heights 1,3,5,7 force the agent to climb each step


class PicoParkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=25):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = 200 # Maximum steps to ensure episodes end

        # Platformer physics
        self.ground_y = 2
        self.JUMP_V0 = 2   # initial vertical velocity on jump
        self.GRAVITY = 2   # vy decreases by this each step → peak jump height = 2 above ground

        # Observations: agent + target positions, current vertical velocity,
        # and a compact encoding of the next obstacle ahead so the policy can react.
        # next_obstacle = [kind, dx, size]
        #   kind: 0 = none (sentinel), 1 = platform, 2 = pit
        #   dx:   tiles from agent to obstacle's left edge (clamped at self.size)
        #   size: platform height OR pit width
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "vy": spaces.Box(-5, self.JUMP_V0, shape=(1,), dtype=np.int32),
                "next_obstacle": spaces.Box(0, size, shape=(3,), dtype=np.int32),
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
            "next_obstacle": self._next_obstacle_ahead(),
        }

    def _generate_obstacles(self):
        # Procedural: chain 2 or 3 challenge templates back-to-back with spacing between them.
        templates = [
            self._challenge_A_ascending,
            self._challenge_B_stairs,
            self._challenge_C_wall_drop,
            self._challenge_D_gap,
            self._challenge_E_tall_wall,
        ]
        n_challenges = int(self.np_random.integers(2, 4))  # 2 or 3
        door_x = self.size - 2
        spacing = 3  # tiles of clear ground after each challenge (room to land + walk)

        obstacles = []
        cur_x = 3  # leave 3 tiles of spawn buffer on the left
        for _ in range(n_challenges):
            # Need ~5 tiles for the widest template + 2 buffer before the door.
            if cur_x + 5 + 2 > door_x:
                break
            template = templates[int(self.np_random.integers(0, len(templates)))]
            chunk = template(cur_x)
            obstacles.extend(chunk)
            max_x = max(o[1] + o[2] - 1 for o in chunk)
            cur_x = max_x + spacing
        return obstacles

    # --- challenge templates ---
    # Each takes start_x and returns a list of (kind, x, w, h) tuples positioned there.
    # Standalone platforms: h <= 2 (jumpable from ground).
    # Stair/ascending platforms: heights step by +2 so each one is only reachable
    #   from the previous step.

    def _challenge_A_ascending(self, start_x):
        # 3 platforms at heights 1, 3, 5 with 1-wide pits in the gaps.
        obstacles = []
        for i in range(3):
            x = start_x + i * 2
            obstacles.append(("platform", x, 1, 1 + 2 * i))
            if i < 2:
                obstacles.append(("pit", x + 1, 1, 0))
        return obstacles

    def _challenge_B_stairs(self, start_x):
        n_steps = int(self.np_random.integers(2, 4))  # 2 or 3
        return [("platform", start_x + i, 1, 1 + 2 * i) for i in range(n_steps)]

    def _challenge_C_wall_drop(self, start_x):
        wall_h = int(self.np_random.integers(1, 3))                  # 1 or 2
        pit_off = int(self.np_random.integers(2, 4))                 # 2 or 3
        pit_w = int(self.np_random.integers(2, 4))                   # 2 or 3
        # Unsolvable combo: pit_off=2 + pit_w=3 leaves the agent no x-gap to land
        # between plat and pit, and the jump-from-plat-top arc lands inside the pit.
        if pit_off == 2 and pit_w == 3:
            pit_w = 2
        return [
            ("platform", start_x, 1, wall_h),
            ("pit", start_x + pit_off, pit_w, 0),
        ]

    def _challenge_D_gap(self, start_x):
        pit_w = int(self.np_random.integers(2, 4))   # 2 or 3
        return [("pit", start_x, pit_w, 0)]

    def _challenge_E_tall_wall(self, start_x):
        return [("platform", start_x, 1, 2)]  # max landable height from ground

    def _next_obstacle_ahead(self):
        agent_x = int(self._agent_location[0])
        best_kind, best_dx, best_size = 0, self.size, 0
        for kind, ox, ow, oh in self._obstacles:
            # Skip obstacles whose right edge is behind the agent.
            if ox + ow - 1 < agent_x:
                continue
            dx = max(0, ox - agent_x)
            if dx < best_dx:
                best_dx = dx
                if kind == "platform":
                    best_kind, best_size = 1, oh
                else:  # pit
                    best_kind, best_size = 2, ow
        return np.array([best_kind, best_dx, best_size], dtype=np.int32)

    def _is_grounded(self, x, y):
        # On floor (and not over a pit)
        if y == self.ground_y:
            for kind, ox, ow, _ in self._obstacles:
                if kind == "pit" and ox <= x <= ox + ow - 1:
                    return False
            return True
        # On top of a platform
        for kind, ox, ow, oh in self._obstacles:
            if kind == "platform" and y == self.ground_y + oh and ox <= x <= ox + ow - 1:
                return True
        return False

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

        # Agent starts in the left spawn buffer; door (target) at far right.
        start_x = self.np_random.integers(0, 3)
        self._agent_location = np.array([start_x, self.ground_y], dtype=np.int32)
        self._target_location = np.array([self.size - 2, self.ground_y], dtype=np.int32)

        # Physics state
        self._vy = 0          # vertical velocity

        # Procedurally generate this episode's obstacles
        self._obstacles = self._generate_obstacles()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1
        prev_location = self._agent_location.copy()
        prev_x, prev_y = int(prev_location[0]), int(prev_location[1])

        h_dir = int(action[0])      # 0=left, 1=none, 2=right
        jump_btn = int(action[1])   # 0/1
        dx = h_dir - 1              # -> -1, 0, +1

        # Jumps allowed when standing on the floor OR on top of a platform (and not over a pit).
        if self._is_grounded(prev_x, prev_y) and jump_btn:
            self._vy = self.JUMP_V0

        # Apply kinematics. Capture cur_vy BEFORE gravity so it reflects this step's motion.
        cur_vy = self._vy
        new_x = int(np.clip(prev_x + dx, 0, self.size - 1))
        new_y = prev_y + cur_vy
        self._vy -= self.GRAVITY

        # Platform side collision: block horizontal motion into the side of a platform.
        for kind, ox, ow, oh in self._obstacles:
            if kind != "platform":
                continue
            plat_top = self.ground_y + oh
            if ox <= new_x <= ox + ow - 1 and self.ground_y <= new_y < plat_top:
                new_x = prev_x
                break

        # Platform landing: if falling onto a platform top, snap y to it.
        for kind, ox, ow, oh in self._obstacles:
            if kind != "platform":
                continue
            plat_top = self.ground_y + oh
            if ox <= new_x <= ox + ow - 1:
                if cur_vy <= 0 and prev_y >= plat_top and new_y <= plat_top:
                    new_y = plat_top
                    self._vy = 0
                    break

        # Phase-through guard: if vy is large-negative the agent can jump sideways from
        # mid-air into a platform with new_y far below the platform's vertical range,
        # so the side check above doesn't fire. Push x back to block the move.
        if new_y < self.ground_y:
            for kind, ox, ow, _ in self._obstacles:
                if kind == "platform" and ox <= new_x <= ox + ow - 1:
                    new_x = prev_x
                    break

        # Pit-aware floor clamp: only clamp at ground_y if NOT over a pit.
        over_pit = any(
            kind == "pit" and ox <= new_x <= ox + ow - 1
            for kind, ox, ow, _ in self._obstacles
        )
        if not over_pit and new_y <= self.ground_y:
            new_y = self.ground_y
            self._vy = 0

        self._agent_location[0] = new_x
        self._agent_location[1] = new_y

        # Out condition: agent fell off the bottom via a pit.
        fell_out = new_y <= 0

        terminated = np.array_equal(self._agent_location, self._target_location) or fell_out
        truncated = self.current_step >= self.max_steps

        # Reward function: shaped progress + small step cost + terminal bonus / penalty
        prev_distance = np.linalg.norm(prev_location - self._target_location, ord=1)
        curr_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)

        reward = 0.1 * (prev_distance - curr_distance) - 0.05
        if np.array_equal(self._agent_location, self._target_location):
            reward += 1.0
        if fell_out:
            reward -= 1.0
        if truncated:
            reward -= 1.0  # Penalize stalling as much as dying so the agent risks jumps.

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

        # Draw obstacles. Pits are white rects punched through the floor; platforms are orange blocks above it.
        orange = (255, 140, 75)
        for kind, ox, ow, oh in self._obstacles:
            if kind == "pit":
                x_px = ox * pix_square_size
                w_px = ow * pix_square_size
                pygame.draw.rect(
                    canvas,
                    (255, 255, 255),
                    pygame.Rect(x_px, floor_top_px, w_px, self.window_size - floor_top_px),
                )
            elif kind == "platform":
                # Platform occupies game-y in [ground_y, ground_y + oh - 1]; top surface at y = ground_y + oh.
                top_row = self.size - (self.ground_y + oh)
                pygame.draw.rect(
                    canvas,
                    orange,
                    pygame.Rect(
                        ox * pix_square_size,
                        top_row * pix_square_size,
                        ow * pix_square_size,
                        oh * pix_square_size,
                    ),
                )

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
