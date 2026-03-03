
import argparse
import json
import os
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRS = [UP, RIGHT, DOWN, LEFT]  # clockwise
DIR2IDX = {d:i for i,d in enumerate(DIRS)}
IDX2DIR = {i:d for i,d in enumerate(DIRS)}

A_STRAIGHT = 0
A_LEFT = 1
A_RIGHT = 2

@dataclass
class StepResult:
    next_state: Tuple[int, ...]
    reward: float
    done: bool
    info: dict

class SnakeEnv:
    def __init__(self, grid: int = 12, seed: int = 0, max_steps: int = 500):
        self.grid = grid
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.snake: Deque[Tuple[int,int]] = deque()
        self.direction = RIGHT
        self.food = (0,0)
        self.steps = 0
        self.score = 0
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        self.steps = 0
        self.score = 0
        cx = self.grid // 2
        cy = self.grid // 2
        self.snake = deque([(cx-1, cy), (cx, cy), (cx+1, cy)])  # tail->head
        self.direction = RIGHT
        self._spawn_food()
        return self._get_state()

    @property
    def head(self):
        return self.snake[-1]

    def _spawn_food(self):
        occ = set(self.snake)
        while True:
            fx = self.rng.randrange(self.grid)
            fy = self.rng.randrange(self.grid)
            if (fx, fy) not in occ:
                self.food = (fx, fy)
                return

    def _is_collision(self, pt: Tuple[int,int]) -> bool:
        x, y = pt
        if x < 0 or x >= self.grid or y < 0 or y >= self.grid:
            return True
        return pt in set(self.snake)

    def _move_dir(self, direction: Tuple[int,int]) -> Tuple[int,int]:
        hx, hy = self.head
        dx, dy = direction
        return (hx + dx, hy + dy)

    def _turn(self, action: int) -> Tuple[int,int]:
        idx = DIR2IDX[self.direction]
        if action == A_STRAIGHT:
            return self.direction
        if action == A_LEFT:
            return IDX2DIR[(idx - 1) % 4]
        if action == A_RIGHT:
            return IDX2DIR[(idx + 1) % 4]
        raise ValueError("Invalid action")

    def _danger(self, direction: Tuple[int,int]) -> int:
        nxt = self._move_dir(direction)
        return int(self._is_collision(nxt))

    def _food_direction_flags(self):
        hx, hy = self.head
        fx, fy = self.food
        food_left = int(fx < hx)
        food_right = int(fx > hx)
        food_up = int(fy < hy)
        food_down = int(fy > hy)
        return food_up, food_down, food_left, food_right

    def _get_state(self) -> Tuple[int, ...]:
        dir_idx = DIR2IDX[self.direction]
        left_dir = IDX2DIR[(dir_idx - 1) % 4]
        right_dir = IDX2DIR[(dir_idx + 1) % 4]

        danger_straight = self._danger(self.direction)
        danger_left = self._danger(left_dir)
        danger_right = self._danger(right_dir)

        food_up, food_down, food_left, food_right = self._food_direction_flags()

        return (danger_straight, danger_left, danger_right,
                food_up, food_down, food_left, food_right,
                dir_idx)

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def step(self, action: int) -> StepResult:
        self.steps += 1
        self.direction = self._turn(action)

        old_dist = self._manhattan(self.head, self.food)

        new_head = self._move_dir(self.direction)
        if self._is_collision(new_head):
            return StepResult(self._get_state(), -10.0, True, {"score": self.score, "reason": "collision"})

        self.snake.append(new_head)

        reward = -0.05
        done = False

        if new_head == self.food:
            self.score += 1
            reward += 10.0
            self._spawn_food()
        else:
            self.snake.popleft()

        new_dist = self._manhattan(self.head, self.food)
        reward += 0.1 if new_dist < old_dist else -0.1

        if self.steps >= self.max_steps:
            done = True

        return StepResult(self._get_state(), reward, done, {"score": self.score})

def epsilon_greedy(qvals: np.ndarray, eps: float, rng: random.Random) -> int:
    if rng.random() < eps:
        return rng.randrange(len(qvals))
    return int(np.argmax(qvals))

def moving_average(x, w=50):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def plot_learning(returns, scores, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(returns, alpha=0.3, label="Return")
    ma = moving_average(returns, w=50)
    if len(ma) > 0:
        plt.plot(np.arange(len(ma)) + 49, ma, label="Return (MA50)")
    plt.xlabel("Episode")
    plt.ylabel("Episodic return")
    plt.title("Learning Curve (Return)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_return.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(scores, alpha=0.3, label="Score")
    ma2 = moving_average(scores, w=50)
    if len(ma2) > 0:
        plt.plot(np.arange(len(ma2)) + 49, ma2, label="Score (MA50)")
    plt.xlabel("Episode")
    plt.ylabel("Score (#foods eaten)")
    plt.title("Learning Curve (Score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_score.png"), dpi=200)
    plt.close()

def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    env = SnakeEnv(grid=args.grid, seed=args.seed, max_steps=args.max_steps)

    Q = defaultdict(lambda: np.zeros(3, dtype=np.float32))
    rng = random.Random(args.seed + 999)
    eps = args.eps_start

    episode_returns = []
    episode_scores = []

    for ep in range(1, args.episodes+1):
        s = env.reset()
        total_r = 0.0

        while True:
            a = epsilon_greedy(Q[s], eps, rng)
            sr = env.step(a)

            best_next = float(np.max(Q[sr.next_state]))
            td_target = sr.reward + args.gamma * best_next * (0.0 if sr.done else 1.0)
            td_error = td_target - Q[s][a]
            Q[s][a] += args.alpha * td_error

            total_r += sr.reward
            s = sr.next_state
            if sr.done:
                episode_returns.append(total_r)
                episode_scores.append(env.score)
                break

        if args.eps_decay_episodes > 0:
            frac = min(1.0, ep / args.eps_decay_episodes)
            eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        else:
            eps = args.eps_end

        if ep % args.log_every == 0:
            print(f"Episode {ep:5d} | eps={eps:.3f} | avg_score(last {args.log_every})={np.mean(episode_scores[-args.log_every:]):.2f}")

    # save Q-table
    q_path = os.path.join(args.out_dir, "q_table.json")
    serial = {",".join(map(str,k)): v.tolist() for k,v in Q.items()}
    with open(q_path, "w") as f:
        json.dump(serial, f)

    plot_learning(episode_returns, episode_scores, args.out_dir)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "episodes": args.episodes,
            "grid": args.grid,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "max_steps": args.max_steps,
            "mean_score_last_200": float(np.mean(episode_scores[-200:])) if len(episode_scores)>=200 else float(np.mean(episode_scores)),
        }, f, indent=2)

    print(f"Saved Q-table: {q_path}")
    print(f"Saved plots/metrics in: {args.out_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_episodes", type=int, default=3000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="artifacts")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
