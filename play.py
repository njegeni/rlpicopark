import gymnasium as gym
import numpy as np
import pygame

import PicoPark_v0  # registers PicoPark-v0


def main():
    env = gym.make("PicoPark-v0", render_mode="human")
    env.reset(seed=0)
    print("Controls: hold A/D or arrows to move, W/up/space to jump, R = reset, ESC = quit")

    pending_jump = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    print("--- reset ---")
                elif event.key in (pygame.K_w, pygame.K_UP, pygame.K_SPACE):
                    pending_jump = True

        if not running:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            h_dir = 0
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            h_dir = 2
        else:
            h_dir = 1
        action = np.array([h_dir, 1 if pending_jump else 0], dtype=np.int64)
        pending_jump = False

        _, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            tag = "TERM" if terminated else "TRUNC"
            print(f"--- {tag} (r={reward:+.2f}) — reset ---")
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
