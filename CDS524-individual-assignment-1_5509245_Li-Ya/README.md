# Snake (Tabular Q-Learning) — Assignment 1

This project implements a **Snake-like** grid game and trains a **tabular Q-learning** agent with an ε-greedy policy.

## Files
- `snake_qlearning.py` : environment + Q-learning training + plotting (no pygame required)
- `play_pygame.py` : pygame visualizer + trained policy playback (run locally)
- `artifacts/` : generated logs/plots/Q-table after training

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python snake_qlearning.py --episodes 4000 --grid 12 --out_dir artifacts
```

## Play (UI demo)
After training:
```bash
python play_pygame.py --grid 12 --q_path artifacts/q_table.json
```

## Notes (important for report)
State is **feature-based** to keep the Q-table small:
- `danger_straight`, `danger_left`, `danger_right` (3 booleans)
- `food_up`, `food_down`, `food_left`, `food_right` (4 booleans)
- `direction` (UP/RIGHT/DOWN/LEFT)

Actions are relative:
- `0=STRAIGHT`, `1=TURN_LEFT`, `2=TURN_RIGHT`

Rewards:
- +10 for eating food
- -10 for dying (wall/self collision)
- -0.05 step penalty (encourage shorter paths)
- +0.1 if moved closer to food, else -0.1 (light shaping)
