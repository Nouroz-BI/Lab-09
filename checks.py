
import torch
import torch.nn.functional as F

from lab09_starter import QNet

# Check 1: DQN target calculation


def check_dqn_targets():
    reward = 1.0
    next_max_q = 7.0
    gamma = 0.9

    # Do not add future Q values after the pole falls.
    terminal_target = reward

    # Add future Q values when the episode has not truly ended.
    nonterminal_target = reward + gamma * next_max_q

    # A time limit does not count as a true ending.
    truncated_target = reward + gamma * next_max_q

    print("\nCHECK 1: DQN TARGETS")
    print("Terminal target:", terminal_target)
    print("Nonterminal target:", nonterminal_target)
    print("Truncated target:", truncated_target)

    assert terminal_target == 1.0
    assert abs(nonterminal_target - 7.3) < 1e-6
    assert abs(truncated_target - 7.3) < 1e-6

    print("DQN target checks passed.")


# Check 2: Huber loss calculation

def check_huber_loss():
    errors = torch.tensor([0.5, 3.0, -3.0])

    # Calculate the Huber loss using its formula.
    manual_loss = torch.where(
        torch.abs(errors) <= 1.0,
        0.5 * errors ** 2,
        torch.abs(errors) - 0.5
    )

    # Calculate the same loss using PyTorch.
    pytorch_loss = F.smooth_l1_loss(
        errors,
        torch.zeros_like(errors),
        reduction="none"
    )

    print("\nCHECK 2: HUBER LOSS")
    print("Manual loss:", manual_loss.tolist())
    print("PyTorch loss:", pytorch_loss.tolist())

    assert torch.allclose(manual_loss, pytorch_loss)

    print("Huber loss checks passed.")


# Check 3: Fixed-point Q-learning

def train_fixed_point(use_target_network):
    torch.manual_seed(0)

    online_net = QNet()

    if use_target_network:
        target_net = QNet()
        target_net.load_state_dict(online_net.state_dict())
    else:
        target_net = None

    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=0.01
    )

    state = torch.zeros((1, 4))

    # Start with the action that has the higher Q value.
    with torch.no_grad():
        action = online_net(state).argmax(dim=1).item()

    reward = 1.0
    gamma = 0.9

    for update in range(3000):
        predicted_q = online_net(state)[0, action]

        with torch.no_grad():
            if use_target_network:
                next_q = target_net(state).max(dim=1)[0][0]
            else:
                next_q = online_net(state).max(dim=1)[0][0]

            target = reward + gamma * next_q

        loss = F.smooth_l1_loss(predicted_q, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Copy the online network every 50 updates.
        if use_target_network and (update + 1) % 50 == 0:
            target_net.load_state_dict(online_net.state_dict())

    with torch.no_grad():
        learned_q = online_net(state)[0, action].item()

    return learned_q



def check_fixed_point():
    target_q = train_fixed_point(use_target_network=True)
    online_q = train_fixed_point(use_target_network=False)

    print("\nCHECK 3: FIXED-POINT Q-LEARNING")
    print("Expected true Q value: 10.0")
    print("With target network:", target_q)
    print("Without target network:", online_q)


# Check 4: PPO clipped objective

def check_ppo_clipping():
    clip_epsilon = 0.2

    examples = [
        (1.3, 1.0),
        (0.7, 1.0),
        (1.3, -1.0),
        (0.7, -1.0)
    ]

    print("\nCHECK 4: PPO CLIPPED OBJECTIVE")

    results = []


    for ratio, advantage in examples:
        unclipped = ratio * advantage

        # Keep the ratio between 0.8 and 1.2.
        clipped_ratio = max(
            1.0 - clip_epsilon,
            min(1.0 + clip_epsilon, ratio)
        )

        clipped = clipped_ratio * advantage

        # PPO uses the smaller of the two values.
        objective = min(unclipped, clipped)

        results.append(objective)

        print(
            f"Ratio: {ratio}, Advantage: {advantage}, "
            f"Unclipped: {unclipped:.2f}, "
            f"Clipped: {clipped:.2f}, "
            f"Objective: {objective:.2f}"
        )

    expected = [1.2, 0.7, -1.3, -0.8]

    assert all(
        abs(actual - target) < 1e-6
        for actual, target in zip(results, expected)
    )

    print("PPO clipping checks passed.")


# Check 5: Return-to-go calculation

def check_return_to_go():
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.5

    returns = []
    running_return = 0.0

    # Calculate returns from the last reward backwards.
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    print("\nCHECK 5: RETURN-TO-GO")
    print("Calculated returns:", returns)

    expected = [1.75, 1.5, 1.0]

    assert all(
        abs(actual - target) < 1e-6
        for actual, target in zip(returns, expected)
    )

    print("Return-to-go checks passed.")


# Run all five checks

if __name__ == "__main__":
    check_dqn_targets()
    check_huber_loss()
    check_fixed_point()
    check_ppo_clipping()
    check_return_to_go()

    print("\nALL CHECKS COMPLETED")
