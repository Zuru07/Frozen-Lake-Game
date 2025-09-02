import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

# ==============================
# Create FrozenLake Environment
# ==============================
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

# ==============================
# Spiking Neural Network
# ==============================
snn = Network()

# Input layer (16 neurons for 16 grid states)
input_layer = Input(n=64, traces=True)

# Hidden layer (64 LIF neurons for learning)
hidden_layer = LIFNodes(n=10,traces=True)

# Connect layers with STDP learning
connection = Connection(
    source=input_layer,
    target=hidden_layer,
    update_rule=PostPre,  # STDP rule
    w=0.5 * torch.rand(64, 10)  # Initialize weights randomly
)

# Add layers and connections to the network
snn.add_layer(input_layer, name="Input")
snn.add_layer(hidden_layer, name="Hidden")
snn.add_connection(connection, source="Input", target="Hidden")

# Add a monitor to observe spiking activity
snn.add_monitor(Monitor(obj=hidden_layer, state_vars=["s"]), name="Hidden_Monitor")

# ==============================
# Training using Q-Learning + SNN
# ==============================
# Q-learning parameters
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
learning_rate = 0.1
q_table = np.zeros((16, 4))  # Q-table (16 states, 4 actions)
num_episodes = 100
reward_log = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Convert state to spike encoding
        spikes = torch.zeros((10,64))
        spikes[:,state] = 1

        # Run network simulation (spike propagation)
        snn.run(inputs={"Input": torch.tensor(spikes, dtype=torch.float)}, time=10)

        # Choose action (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit best Q-value

        # Take action and observe reward
        new_state, reward, done, _, _ = env.step(action)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        # Move to next state
        state = new_state
        total_reward += reward

    # Decay epsilon to reduce exploration over time
    epsilon *= epsilon_decay
    reward_log.append(total_reward)

    # Print progress every 50 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}: Epsilon = {epsilon:.4f}, Reward = {total_reward}")

print("Training completed!")

# ==============================
# Testing Trained Agent
# ==============================
print("Testing trained agent...")
state, _ = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :])  # Choose best action
    new_state, reward, done, _, _ = env.step(action)
    env.render()
    state = new_state

env.close()

# ==============================
# Plot Q-Learning Convergence
# ==============================
plt.plot(range(num_episodes), reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Reward Progress")
plt.grid(True)
plt.show(block=True)