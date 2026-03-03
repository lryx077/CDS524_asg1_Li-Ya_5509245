
import argparse
import json
from collections import defaultdict

import numpy as np
import pygame

from snake_qlearning import SnakeEnv, epsilon_greedy


def load_q(q_path: str):
    with open(q_path, "r") as f:
        raw = json.load(f)
    Q = defaultdict(lambda: np.zeros(3, dtype=np.float32))
    for k, v in raw.items():
        state = tuple(int(x) for x in k.split(","))
        Q[state] = np.array(v, dtype=np.float32)
    return Q


def draw(screen, env: SnakeEnv, cell: int, margin: int, font):
    screen.fill((15, 15, 18))

    # grid lines
    for x in range(env.grid + 1):
        pygame.draw.line(screen, (40, 40, 40), (margin + x * cell, margin), (margin + x * cell, margin + env.grid * cell), 1)
    for y in range(env.grid + 1):
        pygame.draw.line(screen, (40, 40, 40), (margin, margin + y * cell), (margin + env.grid * cell, margin + y * cell), 1)

    # food
    fx, fy = env.food
    pygame.draw.rect(screen, (220, 70, 70), pygame.Rect(margin + fx * cell + 2, margin + fy * cell + 2, cell - 4, cell - 4))

    # snake
    for i, (x, y) in enumerate(env.snake):
        rect = pygame.Rect(margin + x * cell + 2, margin + y * cell + 2, cell - 4, cell - 4)
        color = (60, 200, 110) if i < len(env.snake) - 1 else (90, 240, 150)
        pygame.draw.rect(screen, color, rect)

    txt = font.render(f"Score: {env.score}   Steps: {env.steps}", True, (230, 230, 230))
    screen.blit(txt, (margin, margin + env.grid * cell + 10))


def main(args):
    pygame.init()
    cell = 32
    margin = 20
    info_h = 50
    w = margin * 2 + args.grid * cell
    h = margin * 2 + args.grid * cell + info_h
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Snake Q-learning (Greedy Policy Playback)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    env = SnakeEnv(grid=args.grid, seed=args.seed, max_steps=args.max_steps)
    Q = load_q(args.q_path)

    eps = 0.0
    s = env.reset()
    running = True

    import random
    rng = random.Random(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                s = env.reset()

        a = epsilon_greedy(Q[s], eps, rng)
        sr = env.step(a)
        s = sr.next_state

        if sr.done:
            pygame.time.delay(300)
            s = env.reset()

        draw(screen, env, cell, margin, font)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--q_path", type=str, default="artifacts/q_table.json")
    main(p.parse_args())
