
import random
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=20000):
        # Set the maximum number of transitions the buffer can store.
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state, terminated):
        # Keep the details of each step together.
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(terminated)
        )
        

        # Remove the oldest item if the buffer is already full.
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        # Add the new transition to the buffer.
        self.buffer.append(transition)

    def sample(self, batch_size=64):
       
        # Make sure there are enough transitions for a full batch.
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough transitions to create a batch.")

        # Pick random transitions instead of taking them in order.
        batch = random.sample(self.buffer, batch_size)

        # Separate the transition details into individual groups.
        states, actions, rewards, next_states, terminated = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(terminated, dtype=np.float32)
        )

    def __len__(self):
        # Return the current number of stored transitions.
        return len(self.buffer)


# Test the buffer with a small example.
if __name__ == "__main__":

    random.seed(0)

    # Create a buffer that can hold 100 transitions.
    buffer = ReplayBuffer(capacity=100)

    # Add 100 example transitions to the buffer.
    for i in range(100):
        state = np.array([i, 0, 0, 0], dtype=np.float32)
        action = i % 2
        reward = 1.0
        next_state = state + 1
        terminated = False

        buffer.add(
            state,
            action,
            reward,
            next_state,
            terminated
        )

    # Pick 64 random transitions and check their shapes.
    print("\nREPLAY BUFFER TEST")
    print("Stored transitions:", len(buffer))

    batch = buffer.sample(batch_size=64)

    print("States shape:", batch[0].shape)
    print("Actions shape:", batch[1].shape)
    print("Rewards shape:", batch[2].shape)
    print("Next states shape:", batch[3].shape)
    print("Terminated shape:", batch[4].shape)

    assert len(buffer) == 100
    assert batch[0].shape == (64, 4)
    assert batch[1].shape == (64,)
    assert batch[2].shape == (64,)
    assert batch[3].shape == (64, 4)
    assert batch[4].shape == (64,)

    print("Replay buffer test passed.")
