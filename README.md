# Lab 09 Training an RL Agent

## Introduction

The goal of this lab was to train and compare different reinforcement learning agents using the CartPole environment.

I implemented six agents and tested how they performed over 20,000 environment steps. I also compared how different training settings affected DQN performance, including the use of a target network and replay buffer.

The experiments used three random seeds (0, 1, and 2) so I could compare the results across different runs.


## Project Files

| File | Description |
|---|---|
| `lab09_starter.py` | Original starter code provided for the lab |
| `checks.py` | Checks for important parts of the implementation |
| `replay_buffer.py` | Replay buffer used by DQN |
| `dqn_agent.py` | Standard DQN and its three variations |
| `policy_agents.py` | REINFORCE and PPO implementations |
| `experiments.py` | Runs all experiments and saves the results |
| `requirements.txt` | Required Python packages |
| `.gitignore` | Files and folders excluded from Git |

## Algorithms Implemented

Six agents were tested:

1. **Standard DQN:** Uses a target network and a replay buffer with a capacity of 20,000 transitions.
2. **DQN without a target network:** Uses the online network to calculate the target values.
3. **DQN without a replay buffer:** Uses the most recent 64 transitions in their original order.
4. **DQN with a 1,000-item replay buffer:** Uses a smaller replay buffer instead of the standard 20,000-item buffer.
5. **REINFORCE with a value baseline:** Uses a policy network and a value network to calculate the policy updates.
6. **PPO:** Uses clipped policy updates and Generalized Advantage Estimation (GAE).

## Experimental Setup

All six agents were tested using the same CartPole environment and a budget of 20,000 environment steps per run.

The experiments used the following random seeds:

- Seed 0
- Seed 1
- Seed 2

### DQN settings

The standard DQN used the following settings:

| Setting | Value |
|---|---:|
| Learning rate | 0.001 |
| Discount factor (gamma) | 0.99 |
| Replay buffer capacity | 20,000 |
| Batch size | 64 |
| Learning starts after | 500 steps |
| Target network update | Every 250 steps |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay | First 8,000 steps |
| Loss function | Huber loss |
| Gradient clipping | 10 |

For the DQN variations, I changed only the setting being tested. The smaller replay buffer used a capacity of 1,000, while the version without replay used the latest 64 transitions in order.

### REINFORCE settings

| Setting | Value |
|---|---:|
| Policy learning rate | 0.003 |
| Value learning rate | 0.01 |
| Discount factor | 0.99 |
| Episodes per update | 8 |
| Advantage | Return-to-go minus value estimate |

The advantages were divided by their standard deviation before updating the policy.

### PPO settings

| Setting | Value |
|---|---:|
| Learning rate | 0.0003 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| Clipping value | 0.2 |
| Rollout size | 1,024 |
| Training epochs | 6 |
| Minibatch size | 128 |
| Value loss weight | 0.5 |
| Gradient clipping | 0.5 |

## Implementation Checks

Before running the full experiments, I created and ran `checks.py` to check important parts of the implementation