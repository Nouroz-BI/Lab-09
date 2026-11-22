
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from lab09_starter import (
    CartPole,
    QNet,
    set_seed,
    probe_states
)

from replay_buffer import ReplayBuffer


# Choose an action using epsilon-greedy.
def select_action(network, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(2)

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state_tensor)

    return q_values.argmax(dim=1).item()


# Calculate epsilon based on the current environment step.
def get_epsilon(step):
    start_epsilon = 1.0
    end_epsilon = 0.05
    decay_steps = 8000

    if step >= decay_steps:
        return end_epsilon

    fraction = step / decay_steps

    return start_epsilon + fraction * (
        end_epsilon - start_epsilon
    )


# Train the network using one batch of transitions.
def train_network(
    online_net,
    target_net,
    optimizer,
    batch,
    use_target_network
):
    states, actions, rewards, next_states, terminated = batch

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    terminated = torch.tensor(terminated, dtype=torch.float32)

    # Get the Q value for the action taken.
    current_q = online_net(states).gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # Calculate the target using the next state.
    with torch.no_grad():
        if use_target_network:
            next_q = target_net(next_states).max(dim=1)[0]
        else:
            next_q = online_net(next_states).max(dim=1)[0]

        targets = rewards + 0.99 * (1.0 - terminated) * next_q

    # Calculate the loss and update the network.
    loss = F.smooth_l1_loss(current_q, targets)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        online_net.parameters(),
        max_norm=10.0
    )

    optimizer.step()

    return loss.item()


# Run one DQN experiment.
def run_dqn(
    seed=0,
    variant="standard",
    total_steps=20000
):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Use the settings required for each experiment.
    use_target_network = variant != "no_target"
    use_replay_buffer = variant != "no_replay"

    if variant == "small_buffer":
        buffer_capacity = 1000
    else:
        buffer_capacity = 20000

    # Create the environment and the Q networks.
    env = CartPole()

    online_net = QNet()
    target_net = QNet()

    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=0.001
    )

    # Create the replay buffer.
    if use_replay_buffer:
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity
        )
    else:
        # Keep only the latest 64 transitions in their original order.
        recent_transitions = deque(maxlen=64)

    state = env.reset()

    episode_return = 0.0
    episode_returns = []
    episode_end_steps = []

    q_measurements = []
    q_max_abs_seen = 0.0

    # Use the same fixed states to check Q values.
    fixed_states = np.asarray(
        probe_states(),
        dtype=np.float32
    )

    # Run the agent for the assigned number of environment steps.
    for step in range(1, total_steps + 1):

        epsilon = get_epsilon(step)

        action = select_action(
            online_net,
            state,
            epsilon
        )

        # Keep track of the largest Q value seen during training.
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            action_q_values = online_net(state_tensor)

        current_max_abs = action_q_values.abs().max().item()

        q_max_abs_seen = max(
            q_max_abs_seen,
            current_max_abs
        )

        next_state, reward, terminated, truncated = env.step(action)

        episode_return += reward

        # Store the transition for training.
        transition = (
            state,
            action,
            reward,
            next_state,
            terminated
        )

        if use_replay_buffer:
            replay_buffer.add(*transition)
        else:
            recent_transitions.append(transition)

        state = next_state

        # Train after collecting enough transitions.
        if use_replay_buffer:
            ready_to_train = len(replay_buffer) >= 500
        else:
            ready_to_train = len(recent_transitions) >= 64

        if ready_to_train:

            if use_replay_buffer:
                batch = replay_buffer.sample(64)
            else:
                batch_items = list(recent_transitions)

                states, actions, rewards, next_states, terminated_values = zip(
                    *batch_items
                )

                batch = (
                    np.array(states, dtype=np.float32),
                    np.array(actions, dtype=np.int64),
                    np.array(rewards, dtype=np.float32),
                    np.array(next_states, dtype=np.float32),
                    np.array(terminated_values, dtype=np.float32)
                )

            train_network(
                online_net,
                target_net,
                optimizer,
                batch,
                use_target_network
            )

        # Update the target network every 250 steps.
        if use_target_network and step % 250 == 0:
            target_net.load_state_dict(
                online_net.state_dict()
            )

        # Record Q values every 1000 environment steps.
        if step % 1000 == 0:
            with torch.no_grad():
                probe_tensor = torch.tensor(
                    fixed_states,
                    dtype=torch.float32
                )

                probe_q_values = online_net(probe_tensor)

                mean_max_q = (
                    probe_q_values.max(dim=1)[0].mean().item()
                )

            q_measurements.append({
                "step": step,
                "mean_max_q": mean_max_q,
                "largest_abs_q": q_max_abs_seen
            })

        # End the episode if it terminates or reaches the time limit.
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_end_steps.append(step)

            state = env.reset()
            episode_return = 0.0

    # Keep the last episode return if the step budget ends mid-episode.
    if episode_return > 0:
        episode_returns.append(episode_return)
        episode_end_steps.append(total_steps)

    env.close() if hasattr(env, "close") else None

    return {
        "seed": seed,
        "variant": variant,
        "episode_returns": episode_returns,
        "episode_end_steps": episode_end_steps,
        "q_measurements": q_measurements,
        "largest_abs_q": q_max_abs_seen
    }


# Run this file directly to test one DQN experiment.
if __name__ == "__main__":

    print("\nDQN AGENT TEST")

    results = run_dqn(
        seed=0,
        variant="standard",
        total_steps=20000
    )

    print("Seed:", results["seed"])
    print("Agent:", results["variant"])
    print("Environment steps:", 20000)
    print("Completed episodes:", len(results["episode_returns"]))

    if results["episode_returns"]:
        print(
            "Last episode return:",
            results["episode_returns"][-1]
        )

    print("\nQ-value measurements:")

    for measurement in results["q_measurements"]:
        print(measurement)

    print("\nDQN test completed.")
