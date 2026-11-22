
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from lab09_starter import CartPole, mlp, set_seed

# Choose an action using the policy network.
def select_policy_action(policy, state):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    # Convert the policy output into action probabilities.
    logits = policy(state_tensor)
    distribution = Categorical(logits=logits)

    action = distribution.sample()
    log_probability = distribution.log_prob(action)

    return action.item(), log_probability


# Calculate the return from each step until the end of an episode.
def calculate_returns(rewards, gamma=0.99):
    returns = []
    running_return = 0.0

    # Work backwards through the rewards.
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    return torch.tensor(returns, dtype=torch.float32)


# REINFORCE with a learned value baseline

def run_reinforce(seed=0, total_steps=20000):

    set_seed(seed)

    env = CartPole(seed=seed)

    policy = mlp(4, 2)
    value_network = mlp(4, 1)

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=0.003
    )

    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=0.01
    )

    gamma = 0.99
    episodes_per_update = 8

    episode_returns = []
    episode_end_steps = []

    # Store completed episodes until it is time to update.
    collected_episodes = []

    state = env.reset()
    episode_rewards = []
    episode_states = []
    episode_log_probs = []
    episode_return = 0.0

    environment_steps = 0
    start_time = __import__("time").time()

    while environment_steps < total_steps:

        action, log_probability = select_policy_action(
            policy,
            state
        )

        next_state, reward, terminated, truncated = env.step(action)

        episode_states.append(state)
        episode_rewards.append(reward)
        episode_log_probs.append(log_probability)

        episode_return += reward
        environment_steps += 1

        state = next_state

        # Save the episode when it ends.
        if terminated or truncated:
            collected_episodes.append({
                "states": episode_states,
                "rewards": episode_rewards,
                "log_probs": episode_log_probs
            })

            episode_returns.append(episode_return)
            episode_end_steps.append(environment_steps)

            state = env.reset()
            episode_states = []
            episode_rewards = []
            episode_log_probs = []
            episode_return = 0.0

        # Update the policy after collecting eight episodes.
        if len(collected_episodes) >= episodes_per_update:

            all_states = []
            all_returns = []
            all_log_probs = []

            for episode in collected_episodes:

                returns = calculate_returns(
                    episode["rewards"],
                    gamma
                )

                states = torch.tensor(
                    np.array(episode["states"]),
                    dtype=torch.float32
                )

                with torch.no_grad():
                    baseline = value_network(states).squeeze(1)

                # Compare the actual returns with the value estimate.
                advantages = returns - baseline

                all_states.append(states)
                all_returns.append(returns)
                all_log_probs.extend(episode["log_probs"])

                # Keep the advantages for the policy update.
                if "advantages" not in episode:
                    episode["advantages"] = advantages

            advantages = torch.cat([
                episode["advantages"]
                for episode in collected_episodes
            ])

            # Divide advantages by their standard deviation.
            advantages = advantages / (
                advantages.std(unbiased=False) + 1e-8
            )

            log_probs = torch.stack(all_log_probs)

            # Update the policy network.
            policy_loss = -(
                log_probs * advantages.detach()
            ).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=0.5
            )

            policy_optimizer.step()

            # Update the value network using the episode returns.
            states = torch.cat(all_states)
            returns = torch.cat(all_returns)

            predicted_values = value_network(states).squeeze(1)

            value_loss = F.mse_loss(
                predicted_values,
                returns
            )

            value_optimizer.zero_grad()
            value_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                value_network.parameters(),
                max_norm=0.5
            )

            value_optimizer.step()

            collected_episodes = []

    # Train on any remaining experience collected at the step limit.
    if episode_rewards:
        episode_returns.append(episode_return)
        episode_end_steps.append(environment_steps)

    if collected_episodes:
        all_states = []
        all_returns = []
        all_log_probs = []
        all_advantages = []

        for episode in collected_episodes:
            returns = calculate_returns(episode["rewards"], gamma)

            states = torch.tensor(
                np.array(episode["states"]),
                dtype=torch.float32
            )

            with torch.no_grad():
                baseline = value_network(states).squeeze(1)

            all_states.append(states)
            all_returns.append(returns)
            all_log_probs.extend(episode["log_probs"])
            all_advantages.append(returns - baseline)

        advantages = torch.cat(all_advantages)
        advantages = advantages / (
            advantages.std(unbiased=False) + 1e-8
        )

        log_probs = torch.stack(all_log_probs)

        policy_loss = -(
            log_probs * advantages.detach()
        ).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=0.5
        )

        policy_optimizer.step()

        states = torch.cat(all_states)
        returns = torch.cat(all_returns)

        value_loss = F.mse_loss(
            value_network(states).squeeze(1),
            returns
        )

        value_optimizer.zero_grad()
        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            value_network.parameters(),
            max_norm=0.5
        )

        value_optimizer.step()

    elapsed_time = __import__("time").time() - start_time

    return {
        "seed": seed,
        "agent": "REINFORCE",
        "episode_returns": episode_returns,
        "episode_end_steps": episode_end_steps,
        "environment_steps": environment_steps,
        "elapsed_time": elapsed_time
    }


# Calculate advantages for PPO

def calculate_gae(
    rewards,
    values,
    next_values,
    terminated,
    episode_ends,
    gamma=0.99,
    gae_lambda=0.95
):
    advantages = []
    running_advantage = 0.0

    # Calculate GAE from the last step backwards.
    for i in reversed(range(len(rewards))):

        delta = (
            rewards[i]
            + gamma * next_values[i] * (1.0 - terminated[i])
            - values[i]
        )

        running_advantage = (
            delta
            + gamma
            * gae_lambda
            * (1.0 - episode_ends[i])
            * running_advantage
        )

        advantages.insert(0, running_advantage)

    return torch.tensor(advantages, dtype=torch.float32)


# PPO training

def run_ppo(seed=0, total_steps=20000):

    set_seed(seed)

    env = CartPole(seed=seed)

    policy = mlp(4, 2)
    value_network = mlp(4, 1)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_network.parameters()),
        lr=0.0003
    )

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    rollout_size = 1024
    epochs = 6
    batch_size = 128

    episode_returns = []
    episode_end_steps = []

    state = env.reset()
    episode_return = 0.0
    environment_steps = 0

    start_time = __import__("time").time()

    while environment_steps < total_steps:

        # Store one rollout before updating the networks.
        states = []
        actions = []
        rewards = []
        next_states = []
        terminated_values = []
        episode_ends = []
        old_log_probs = []

        steps_in_rollout = min(
            rollout_size,
            total_steps - environment_steps
        )

        for _ in range(steps_in_rollout):

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            logits = policy(state_tensor)
            distribution = Categorical(logits=logits)

            action_tensor = distribution.sample()
            log_probability = distribution.log_prob(action_tensor)

            action = action_tensor.item()

            next_state, reward, terminated, truncated = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminated_values.append(float(terminated))
            episode_ends.append(float(terminated or truncated))
            old_log_probs.append(log_probability.item())

            episode_return += reward
            environment_steps += 1

            state = next_state

            # Start a new episode when the current one ends.
            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_end_steps.append(environment_steps)

                state = env.reset()
                episode_return = 0.0


        # Convert the rollout data into tensors.
        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        terminated_values = torch.tensor(
            terminated_values,
            dtype=torch.float32
        )

        episode_ends = torch.tensor(
            episode_ends,
            dtype=torch.float32
        )

        old_log_probs = torch.tensor(
            old_log_probs,
            dtype=torch.float32
        )

        # Estimate the value of each state and next state.
        with torch.no_grad():
            values = value_network(states).squeeze(1)
            next_values = value_network(next_states).squeeze(1)

        # Calculate advantages and target returns.
        advantages = calculate_gae(
            rewards.tolist(),
            values.tolist(),
            next_values.tolist(),
            terminated_values.tolist(),
            episode_ends.tolist(),
            gamma,
            gae_lambda
        )

        returns = advantages + values

        # Normalize advantages before updating the policy.
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        # Train the policy and value networks for six epochs.
        for _ in range(epochs):

            indices = torch.randperm(len(states))

            for start in range(0, len(states), batch_size):

                batch_indices = indices[
                    start:start + batch_size
                ]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits = policy(batch_states)
                distribution = Categorical(logits=logits)

                new_log_probs = distribution.log_prob(
                    batch_actions
                )

                # Compare the new action probabilities with the old ones.
                ratios = torch.exp(
                    new_log_probs - batch_old_log_probs
                )

                unclipped = ratios * batch_advantages

                clipped = torch.clamp(
                    ratios,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon
                ) * batch_advantages

                policy_loss = -torch.minimum(
                    unclipped,
                    clipped
                ).mean()

                # Train the value network to predict the returns.
                predicted_values = value_network(
                    batch_states
                ).squeeze(1)

                value_loss = F.mse_loss(
                    predicted_values,
                    batch_returns
                )

                # Combine the policy loss and value loss.
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters())
                    + list(value_network.parameters()),
                    max_norm=0.5
                )

                optimizer.step()

    # Record the final episode if the step budget ends mid-episode.
    if episode_return > 0:
        episode_returns.append(episode_return)
        episode_end_steps.append(environment_steps)

    elapsed_time = __import__("time").time() - start_time

    return {
        "seed": seed,
        "agent": "PPO",
        "episode_returns": episode_returns,
        "episode_end_steps": episode_end_steps,
        "environment_steps": environment_steps,
        "elapsed_time": elapsed_time
    }


# Run a short test of both methods.
if __name__ == "__main__":

    print("\nREINFORCE TEST")

    reinforce_results = run_reinforce(
        seed=0,
        total_steps=20000
    )

    print("Environment steps:", reinforce_results["environment_steps"])
    print("Completed episodes:", len(reinforce_results["episode_returns"]))
    print("Last episode return:", reinforce_results["episode_returns"][-1])
    print("Training time:", round(reinforce_results["elapsed_time"], 2), "seconds")

    print("\nPPO TEST")

    ppo_results = run_ppo(
        seed=0,
        total_steps=20000
    )

    print("Environment steps:", ppo_results["environment_steps"])

    print("Completed episodes:", len(ppo_results["episode_returns"]))
    
    print("Last episode return:", ppo_results["episode_returns"][-1])
    
    print("Training time:", round(ppo_results["elapsed_time"], 2), "seconds")

    print("\nBOTH POLICY AGENTS COMPLETED")
